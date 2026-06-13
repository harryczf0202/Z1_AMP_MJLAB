import os
import inspect
import math
import types
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import wandb
import rsl_rl
import amp_rsl_rl

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.utils.lab_api.math import quat_apply_inverse
from amp_rsl_rl.runners.amp_on_policy_runner import AMPOnPolicyRunner as BaseAMPOnPolicyRunner
from amp_rsl_rl.runners.amp_on_policy_runner import resolve_class
from amp_rsl_rl.utils._compat import RSL_RL_V4_PLUS, RSL_RL_V5_PLUS, resolve_obs_groups
from amp_rsl_rl.algorithms import AMP_PPO
from amp_rsl_rl.networks import Discriminator

from .body_amp_loader import BodyAMPLoader


class _OnnxPolicyWrapper(torch.nn.Module):
  """Thin wrapper that exposes ``act_inference`` as ``forward`` for ONNX export.
  
  Includes the obs normalizer so the exported ONNX model expects raw observations
  and C++ deployment does not need to implement normalization separately.
  """

  def __init__(self, actor_critic, obs_normalizer=None):
    super().__init__()
    self.actor_critic = actor_critic
    self.obs_normalizer = obs_normalizer

  def forward(self, obs):
    if self.obs_normalizer is not None:
      obs = self.obs_normalizer(obs)
    return self.actor_critic.act_inference(obs)


def _onnx_export_kwargs_single_file() -> dict:
  """Build kwargs that request single-file ONNX export across torch versions."""
  try:
    params = inspect.signature(torch.onnx.export).parameters
  except (TypeError, ValueError):
    return {}

  if "external_data" in params:
    return {"external_data": False}
  if "use_external_data_format" in params:
    return {"use_external_data_format": False}
  return {}


def _inline_external_onnx_data(onnx_path: str) -> None:
  """Merge external tensor data back into a single ONNX file if needed."""
  data_path = f"{onnx_path}.data"
  if not os.path.exists(data_path):
    return

  try:
    import onnx

    model = onnx.load(onnx_path, load_external_data=True)
    onnx.save_model(model, onnx_path, save_as_external_data=False)
    if os.path.exists(data_path):
      os.remove(data_path)
    print(f"[INFO]: Inlined external ONNX data into single file: {onnx_path}")
  except Exception as exc:
    print(f"[WARN]: Failed to inline ONNX external data for {onnx_path}: {exc}")


def _safe_nan_to_num(tensor: torch.Tensor) -> torch.Tensor:
  return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def _finite_row_mask(*tensors: torch.Tensor) -> torch.Tensor:
  mask = torch.ones(tensors[0].shape[0], dtype=torch.bool, device=tensors[0].device)
  for tensor in tensors:
    flat = tensor.reshape(tensor.shape[0], -1)
    mask &= torch.isfinite(flat).all(dim=1)
  return mask


class SafeAMP_PPO(AMP_PPO):
  """AMP PPO with local non-finite guards."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.last_safety_stats: dict[str, float] = {
      "policy_samples_dropped": 0.0,
      "expert_samples_dropped": 0.0,
      "invalid_minibatches": 0.0,
    }

  def _sample_rows(self, tensor: torch.Tensor, count: int) -> torch.Tensor:
    if tensor.shape[0] == count:
      return tensor
    indices = torch.randperm(tensor.shape[0], device=tensor.device)[:count]
    return tensor[indices]

  def update(
    self,
  ) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
    mean_value_loss: float = 0.0
    mean_surrogate_loss: float = 0.0
    mean_amp_loss: float = 0.0
    mean_grad_pen_loss: float = 0.0
    mean_policy_pred: float = 0.0
    mean_expert_pred: float = 0.0
    mean_accuracy_policy: float = 0.0
    mean_accuracy_expert: float = 0.0
    mean_accuracy_policy_elem: float = 0.0
    mean_accuracy_expert_elem: float = 0.0
    mean_kl_divergence: float = 0.0
    mean_symmetry_loss: float = 0.0

    dropped_policy_samples = 0.0
    dropped_expert_samples = 0.0
    invalid_minibatches = 0.0

    _is_recurrent = self.actor.is_recurrent if RSL_RL_V4_PLUS else self.actor_critic.is_recurrent

    if _is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )
    else:
      generator = self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )

    amp_policy_generator = self.amp_storage.feed_forward_generator(
      num_mini_batch=self.num_learning_epochs * self.num_mini_batches,
      mini_batch_size=(
        self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
      ),
      allow_replacement=True,
    )
    amp_expert_generator = self.amp_data.feed_forward_generator(
      self.num_learning_epochs * self.num_mini_batches,
      self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
    )

    for sample, sample_amp_policy, sample_amp_expert in zip(
      generator, amp_policy_generator, amp_expert_generator
    ):
      if hasattr(sample, "observations"):
        obs_batch = sample.observations
        actions_batch = sample.actions
        target_values_batch = sample.values
        advantages_batch = sample.advantages
        returns_batch = sample.returns
        old_actions_log_prob_batch = sample.old_actions_log_prob
        old_mu_batch = sample.old_distribution_params[0]
        old_sigma_batch = sample.old_distribution_params[1]
        hidden_states_batch = sample.hidden_states
        masks_batch = sample.masks
      elif isinstance(sample, tuple) and len(sample) == 9:
        (
          obs_batch,
          actions_batch,
          target_values_batch,
          advantages_batch,
          returns_batch,
          old_actions_log_prob_batch,
          old_distribution_params_batch,
          hidden_states_batch,
          masks_batch,
        ) = sample
        old_mu_batch = old_distribution_params_batch[0]
        old_sigma_batch = old_distribution_params_batch[1]
      else:
        (
          obs_batch,
          actions_batch,
          target_values_batch,
          advantages_batch,
          returns_batch,
          old_actions_log_prob_batch,
          old_mu_batch,
          old_sigma_batch,
          hidden_states_batch,
          masks_batch,
        ) = sample

      hidden_state_actor, hidden_state_critic = (None, None)
      if hidden_states_batch is not None:
        hidden_state_actor, hidden_state_critic = hidden_states_batch

      original_batch_size = obs_batch.shape[0]

      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          advantages_batch = (advantages_batch - advantages_batch.mean()) / (
            advantages_batch.std() + 1e-8
          )

      if self.symmetry_cfg and self.symmetry_cfg.get("use_data_augmentation", False):
        aug_obs, aug_actions = self._apply_symmetry(
          obs=obs_batch,
          actions=actions_batch,
          obs_type=["policy", "critic"],
        )
        num_aug = self._augment_batch_size(original_batch_size, aug_obs)
        obs_batch = aug_obs
        actions_batch = aug_actions

        old_actions_log_prob_batch = self._repeat_along_batch(old_actions_log_prob_batch, num_aug)
        target_values_batch = self._repeat_along_batch(target_values_batch, num_aug)
        advantages_batch = self._repeat_along_batch(advantages_batch, num_aug)
        returns_batch = self._repeat_along_batch(returns_batch, num_aug)

      if RSL_RL_V4_PLUS:
        _ = self.actor(
          obs_batch,
          masks=masks_batch,
          hidden_state=hidden_state_actor,
          stochastic_output=True,
        )
        actions_log_prob_batch = self.actor.get_output_log_prob(actions_batch)
        value_batch = self.critic(
          obs_batch, masks=masks_batch, hidden_state=hidden_state_critic
        )
        if hasattr(self.actor, "get_distribution_params"):
          dist_params = self.actor.get_distribution_params()
          mu_batch = dist_params[0][:original_batch_size]
          sigma_batch = dist_params[1][:original_batch_size]
          entropy_batch = self.actor.get_entropy()[:original_batch_size]
        else:
          mu_batch = self.actor.output_mean[:original_batch_size]
          sigma_batch = self.actor.output_std[:original_batch_size]
          entropy_batch = self.actor.output_entropy[:original_batch_size]
      else:
        self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hidden_state_actor)
        actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
        value_batch = self.actor_critic.evaluate(
          obs_batch, masks=masks_batch, hidden_states=hidden_state_critic
        )
        mu_batch = self.actor_critic.action_mean[:original_batch_size]
        sigma_batch = self.actor_critic.action_std[:original_batch_size]
        entropy_batch = self.actor_critic.entropy[:original_batch_size]

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = torch.sum(
            torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
            + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
            / (2.0 * torch.square(sigma_batch))
            - 0.5,
            axis=-1,
          )
          kl_mean = torch.mean(kl)
          mean_kl_divergence += kl_mean.item()

          if kl_mean > self.desired_kl * 2.0:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
          elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

          for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

      ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))

      min_ = 1.0 - self.clip_param
      max_ = 1.0 + self.clip_param
      if self.use_smooth_ratio_clipping:
        clipped_ratio = (
          1 / (1 + torch.exp((-(ratio - min_) / (max_ - min_) + 0.5) * 4))
          * (max_ - min_)
          + min_
        )
      else:
        clipped_ratio = torch.clamp(ratio, min_, max_)

      surrogate = -torch.squeeze(advantages_batch) * ratio
      surrogate_clipped = -torch.squeeze(advantages_batch) * clipped_ratio
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = target_values_batch + (
          value_batch - target_values_batch
        ).clamp(-self.clip_param, self.clip_param)
        value_losses = (value_batch - returns_batch).pow(2)
        value_losses_clipped = (value_clipped - returns_batch).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (returns_batch - value_batch).pow(2).mean()

      ppo_loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy_batch.mean()
      )
      ppo_loss = _safe_nan_to_num(ppo_loss)
      surrogate_loss = _safe_nan_to_num(surrogate_loss)
      value_loss = _safe_nan_to_num(value_loss)

      symmetry_loss_value = torch.zeros(1, device=self.device)
      if self.symmetry_cfg:
        if not self.symmetry_cfg.get("use_data_augmentation", False):
          sym_obs_batch, _ = self._apply_symmetry(
            obs=obs_batch[:original_batch_size],
            actions=None,
            obs_type="policy",
          )
        else:
          sym_obs_batch = obs_batch

        if sym_obs_batch is not None:
          with torch.no_grad():
            sym_obs_detached = sym_obs_batch.detach().clone()
          if RSL_RL_V4_PLUS:
            mean_actions_batch = self.actor(sym_obs_detached, stochastic_output=False)
          else:
            mean_actions_batch = self.actor_critic.act_inference(sym_obs_detached)
          action_mean_orig = mean_actions_batch[:original_batch_size]
          _, sym_actions = self._apply_symmetry(
            obs=None,
            actions=action_mean_orig,
            obs_type="policy",
          )
          if sym_actions is None:
            sym_actions = mean_actions_batch
          mse_loss = torch.nn.MSELoss()
          symmetry_loss_value = mse_loss(
            mean_actions_batch[original_batch_size:],
            sym_actions.detach()[original_batch_size:],
          )
          if self.symmetry_cfg.get("use_mirror_loss", False):
            coeff = self.symmetry_cfg.get("mirror_loss_coeff", 0.0)
            ppo_loss = ppo_loss + coeff * symmetry_loss_value
          else:
            symmetry_loss_value = symmetry_loss_value.detach()

      policy_state, policy_next_state = sample_amp_policy
      expert_state, expert_next_state = sample_amp_expert

      if self.symmetry_cfg and self.symmetry_cfg.get("use_data_augmentation", False):
        policy_state = self.discriminator.apply_symmetry(policy_state, obs_type="amp")
        policy_next_state = self.discriminator.apply_symmetry(policy_next_state, obs_type="amp")
        expert_state = self.discriminator.apply_symmetry(expert_state, obs_type="amp")
        expert_next_state = self.discriminator.apply_symmetry(expert_next_state, obs_type="amp")

      policy_state = policy_state.to(self.device)
      policy_next_state = policy_next_state.to(self.device)
      expert_state = expert_state.to(self.device)
      expert_next_state = expert_next_state.to(self.device)

      policy_valid = _finite_row_mask(policy_state, policy_next_state)
      expert_valid = _finite_row_mask(expert_state, expert_next_state)
      dropped_policy_samples += float((~policy_valid).sum().item())
      dropped_expert_samples += float((~expert_valid).sum().item())

      policy_state = _safe_nan_to_num(policy_state[policy_valid])
      policy_next_state = _safe_nan_to_num(policy_next_state[policy_valid])
      expert_state = _safe_nan_to_num(expert_state[expert_valid])
      expert_next_state = _safe_nan_to_num(expert_next_state[expert_valid])

      amp_loss = torch.zeros(1, device=self.device, dtype=ppo_loss.dtype)
      grad_pen_loss = torch.zeros(1, device=self.device, dtype=ppo_loss.dtype)
      policy_d_prob = torch.zeros(0, device=self.device)
      expert_d_prob = torch.zeros(0, device=self.device)

      valid_amp_count = min(policy_state.shape[0], expert_state.shape[0])
      if valid_amp_count <= 0:
        invalid_minibatches += 1.0
      else:
        policy_state = self._sample_rows(policy_state, valid_amp_count)
        policy_next_state = self._sample_rows(policy_next_state, valid_amp_count)
        expert_state = self._sample_rows(expert_state, valid_amp_count)
        expert_next_state = self._sample_rows(expert_next_state, valid_amp_count)

        policy_state_raw = policy_state.detach().clone()
        policy_next_state_raw = policy_next_state.detach().clone()
        expert_state_raw = expert_state.detach().clone()
        expert_next_state_raw = expert_next_state.detach().clone()

        discriminator_input = torch.cat(
          (
            torch.cat([policy_state, policy_next_state], dim=-1),
            torch.cat([expert_state, expert_next_state], dim=-1),
          ),
          dim=0,
        )
        discriminator_input = _safe_nan_to_num(discriminator_input)
        discriminator_output = self.discriminator(discriminator_input)
        discriminator_output = _safe_nan_to_num(discriminator_output)

        policy_d = discriminator_output[:valid_amp_count]
        expert_d = discriminator_output[valid_amp_count:]

        amp_loss, grad_pen_loss = self.discriminator.compute_loss(
          policy_d=policy_d,
          expert_d=expert_d,
          sample_amp_expert=(expert_state, expert_next_state),
          sample_amp_policy=(policy_state, policy_next_state),
          lambda_=10,
        )
        amp_loss = _safe_nan_to_num(amp_loss)
        grad_pen_loss = _safe_nan_to_num(grad_pen_loss)

        if not torch.isfinite(amp_loss).all() or not torch.isfinite(grad_pen_loss).all():
          amp_loss = torch.zeros(1, device=self.device, dtype=ppo_loss.dtype)
          grad_pen_loss = torch.zeros(1, device=self.device, dtype=ppo_loss.dtype)
          invalid_minibatches += 1.0
        else:
          self.discriminator.update_normalization(
            expert_state_raw,
            expert_next_state_raw,
            policy_state_raw,
            policy_next_state_raw,
          )

          policy_d_prob = torch.sigmoid(_safe_nan_to_num(policy_d))
          expert_d_prob = torch.sigmoid(_safe_nan_to_num(expert_d))

      loss = ppo_loss + amp_loss + grad_pen_loss
      loss = _safe_nan_to_num(loss)

      self.optimizer.zero_grad()
      loss.backward()
      if RSL_RL_V4_PLUS:
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      else:
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
      self.optimizer.step()

      mean_value_loss += float(value_loss.item())
      mean_surrogate_loss += float(surrogate_loss.item())
      mean_amp_loss += float(amp_loss.item())
      mean_grad_pen_loss += float(grad_pen_loss.item())
      mean_policy_pred += float(policy_d_prob.mean().item()) if policy_d_prob.numel() > 0 else 0.0
      mean_expert_pred += float(expert_d_prob.mean().item()) if expert_d_prob.numel() > 0 else 0.0
      mean_symmetry_loss += float(symmetry_loss_value.item())

      if policy_d_prob.numel() > 0:
        mean_accuracy_policy += torch.sum(
          torch.round(policy_d_prob) == torch.zeros_like(policy_d_prob)
        ).item()
        mean_accuracy_policy_elem += float(policy_d_prob.numel())
      if expert_d_prob.numel() > 0:
        mean_accuracy_expert += torch.sum(
          torch.round(expert_d_prob) == torch.ones_like(expert_d_prob)
        ).item()
        mean_accuracy_expert_elem += float(expert_d_prob.numel())

    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_amp_loss /= num_updates
    mean_grad_pen_loss /= num_updates
    mean_policy_pred /= num_updates
    mean_expert_pred /= num_updates
    mean_accuracy_policy /= max(1.0, mean_accuracy_policy_elem)
    mean_accuracy_expert /= max(1.0, mean_accuracy_expert_elem)
    mean_kl_divergence /= num_updates
    mean_symmetry_loss /= num_updates

    self.storage.clear()
    self.last_safety_stats = {
      "policy_samples_dropped": dropped_policy_samples,
      "expert_samples_dropped": dropped_expert_samples,
      "invalid_minibatches": invalid_minibatches,
    }

    return (
      mean_value_loss,
      mean_surrogate_loss,
      mean_amp_loss,
      mean_grad_pen_loss,
      mean_policy_pred,
      mean_expert_pred,
      mean_accuracy_policy,
      mean_accuracy_expert,
      mean_kl_divergence,
      mean_symmetry_loss,
    )


class AMPOnPolicyRunner(BaseAMPOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
    self.cfg = train_cfg
    self.alg_cfg = train_cfg["algorithm"]
    self.discriminator_cfg = train_cfg["discriminator"]
    self.dataset_cfg = train_cfg["dataset"]
    self.device = device
    self.env = env
    self.style_weight = train_cfg.get("style_weight", 0.5)
    self._export_policy_fn = None

    observations = self.env.get_observations()
    default_sets = ["critic"]
    self.cfg["obs_groups"] = resolve_obs_groups(
      observations, self.cfg.get("obs_groups"), default_sets
    )

    if RSL_RL_V4_PLUS:
      self.actor_cfg = train_cfg["actor"]
      self.critic_cfg = train_cfg["critic"]
      self.policy_cfg = {"actor": self.actor_cfg, "critic": self.critic_cfg}

      actor_class = resolve_class(self.actor_cfg.pop("class_name", "MLPModel"))
      critic_class = resolve_class(self.critic_cfg.pop("class_name", "MLPModel"))

      actor_valid_keys = set(inspect.signature(actor_class.__init__).parameters.keys())
      critic_valid_keys = set(inspect.signature(critic_class.__init__).parameters.keys())
      self.actor_cfg = {k: v for k, v in self.actor_cfg.items() if k in actor_valid_keys}
      self.critic_cfg = {k: v for k, v in self.critic_cfg.items() if k in critic_valid_keys}

      actor = actor_class(
        observations,
        self.cfg["obs_groups"],
        "actor",
        self.env.num_actions,
        **self.actor_cfg,
      ).to(self.device)
      critic = critic_class(
        observations,
        self.cfg["obs_groups"],
        "critic",
        1,
        **self.critic_cfg,
      ).to(self.device)
    else:
      raise RuntimeError("Z1-AMP-MJLAB currently supports only rsl_rl v4-style configs.")

    sim_cfg = getattr(self.env.cfg, "sim", None)
    if sim_cfg is None or not hasattr(sim_cfg, "dt"):
      raise AttributeError(
        "env.cfg.sim.dt is not set. Please ensure your environment config defines `sim.dt`."
      )
    if not hasattr(self.env.cfg, "decimation"):
      raise AttributeError(
        "env.cfg.decimation is not set. Please ensure your environment config defines `decimation`."
      )

    num_amp_obs = self._flatten_amp_obs(observations["amp"]).shape[1]
    amp_data = BodyAMPLoader(
      motion_dir=self.dataset_cfg["amp_data_path"],
      body_names=self.dataset_cfg["amp_body_names"],
      anchor_name=self.dataset_cfg["amp_anchor_name"],
      joint_names=self.dataset_cfg.get("amp_joint_names"),
      device=self.device,
      group_weights=self.dataset_cfg.get("group_weights"),
    )
    if amp_data.all_obs.shape[1] != num_amp_obs:
      raise ValueError(
        f"Body AMP expert observation dim {amp_data.all_obs.shape[1]} does not match env AMP dim {num_amp_obs}."
      )

    self.discriminator = Discriminator(
      input_dim=num_amp_obs * 2,
      hidden_layer_sizes=self.discriminator_cfg["hidden_dims"],
      reward_scale=self.discriminator_cfg["reward_scale"],
      device=self.device,
      loss_type=self.discriminator_cfg["loss_type"],
      empirical_normalization=self.discriminator_cfg["empirical_normalization"],
      symmetry_cfg=self.alg_cfg.get("symmetry_cfg"),
    ).to(self.device)

    alg_cfg = dict(self.alg_cfg)
    alg_class = resolve_class(alg_cfg.pop("class_name"))
    for key in list(alg_cfg.keys()):
      if key not in AMP_PPO.__init__.__code__.co_varnames:
        alg_cfg.pop(key)

    alg_impl_class = SafeAMP_PPO if issubclass(alg_class, AMP_PPO) else alg_class

    self.alg = alg_impl_class(
      actor=actor,
      critic=critic,
      discriminator=self.discriminator,
      amp_data=amp_data,
      device=self.device,
      **alg_cfg,
    )

    self.num_steps_per_env = self.cfg["num_steps_per_env"]
    self.save_interval = self.cfg["save_interval"]

    obs_template = observations.clone().detach().to(self.device)
    self.alg.init_storage(
      self.env.num_envs,
      self.num_steps_per_env,
      obs_template,
      (self.env.num_actions,),
    )

    self.empirical_normalization = bool(
      train_cfg.get("empirical_normalization", False)
      or train_cfg["actor"].get("obs_normalization", False)
    )
    self.log_dir = log_dir
    self.logger_type = None
    self.tot_timesteps = 0
    self.tot_time = 0
    self.current_learning_iteration = 0

    self.git_status_repos = [rsl_rl.__file__, amp_rsl_rl.__file__]
    self.writer = None
    self._rollout_safety_stats = {
      "amp_sanitized_envs": 0.0,
      "style_reward_sanitized_envs": 0.0,
    }
    self._sync_policy_std_compat()
    self._install_policy_safety_hooks()
    self._install_amp_safety_hooks()

  def _sync_policy_std_compat(self) -> None:
    """Expose compatibility attributes expected by upstream AMP logging helpers."""
    actor = getattr(self.alg, "actor", None)
    if actor is None or not hasattr(actor, "distribution"):
      return

    dist = actor.distribution
    if hasattr(dist, "log_std_param"):
      actor.noise_std_type = "log"
      actor.log_std = dist.log_std_param
    elif hasattr(dist, "std_param"):
      actor.noise_std_type = "scalar"
      actor.std = dist.std_param
    else:
      actor.noise_std_type = "scalar"

  def _clamp_policy_std_(self) -> None:
    """Keep policy exploration std in a sane positive range."""
    actor = getattr(self.alg, "actor", None)
    if actor is None or not hasattr(actor, "distribution"):
      return

    dist = actor.distribution
    min_std = torch.tensor(
      self.cfg.get("min_normalized_std", [0.05] * self.env.num_actions),
      device=self.device,
      dtype=torch.float32,
    )
    max_std = torch.tensor(
      self.cfg.get("max_normalized_std", [1.0] * self.env.num_actions),
      device=self.device,
      dtype=torch.float32,
    )

    with torch.no_grad():
      if hasattr(dist, "log_std_param"):
        min_log = torch.log(min_std)
        max_log = torch.log(max_std)
        safe_log_std = torch.nan_to_num(
          dist.log_std_param.data,
          nan=math.log(0.2),
          posinf=max_log.max().item(),
          neginf=min_log.min().item(),
        )
        dist.log_std_param.data.copy_(safe_log_std.clamp(min=min_log, max=max_log))
      elif hasattr(dist, "std_param"):
        safe_std = torch.nan_to_num(
          dist.std_param.data,
          nan=0.2,
          posinf=max_std.max().item(),
          neginf=min_std.min().item(),
        )
        dist.std_param.data.copy_(safe_std.clamp(min=min_std, max=max_std))
    self._sync_policy_std_compat()

  def _install_policy_safety_hooks(self) -> None:
    """Sanitize policy distribution parameters before sampling and after optimizer steps."""
    actor = getattr(self.alg, "actor", None)
    if actor is None or not hasattr(actor, "distribution"):
      return

    dist = actor.distribution
    if not getattr(dist, "_z1_safe_update_installed", False):
      def _safe_update(distribution_self, mlp_output: torch.Tensor) -> None:
        mean = torch.nan_to_num(mlp_output, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
        min_std = torch.tensor(
          self.cfg.get("min_normalized_std", [0.05] * self.env.num_actions),
          device=self.device,
          dtype=mean.dtype,
        )
        max_std = torch.tensor(
          self.cfg.get("max_normalized_std", [1.0] * self.env.num_actions),
          device=self.device,
          dtype=mean.dtype,
        )

        with torch.no_grad():
          if hasattr(distribution_self, "log_std_param"):
            min_log = torch.log(min_std)
            max_log = torch.log(max_std)
            safe_log_std = torch.nan_to_num(
              distribution_self.log_std_param.data,
              nan=math.log(0.2),
              posinf=max_log.max().item(),
              neginf=min_log.min().item(),
            )
            distribution_self.log_std_param.data.copy_(safe_log_std.clamp(min=min_log, max=max_log))
            std = torch.exp(distribution_self.log_std_param).expand_as(mean)
          elif hasattr(distribution_self, "std_param"):
            safe_std = torch.nan_to_num(
              distribution_self.std_param.data,
              nan=0.2,
              posinf=max_std.max().item(),
              neginf=min_std.min().item(),
            )
            distribution_self.std_param.data.copy_(safe_std.clamp(min=min_std, max=max_std))
            std = distribution_self.std_param.expand_as(mean)
          else:
            raise AttributeError("Unsupported policy distribution without std parameters.")

        distribution_self._distribution = torch.distributions.Normal(mean, std)

      dist.update = types.MethodType(_safe_update, dist)
      dist._z1_safe_update_installed = True

    if not getattr(self.alg.optimizer, "_z1_safe_step_installed", False):
      orig_step = self.alg.optimizer.step

      def _safe_step(*args, **kwargs):
        result = orig_step(*args, **kwargs)
        self._clamp_policy_std_()
        return result

      self.alg.optimizer.step = _safe_step
      self.alg.optimizer._z1_safe_step_installed = True

  def _reset_amp_safety_stats(self) -> None:
    for key in self._rollout_safety_stats:
      self._rollout_safety_stats[key] = 0.0
    if hasattr(self.alg, "last_safety_stats"):
      self.alg.last_safety_stats = {
        "policy_samples_dropped": 0.0,
        "expert_samples_dropped": 0.0,
        "invalid_minibatches": 0.0,
      }



  def _install_amp_safety_hooks(self) -> None:
    if getattr(self.alg, "_z1_amp_safety_hooks_installed", False):
      return

    original_act_amp = self.alg.act_amp
    original_process_amp_step = self.alg.process_amp_step
    original_predict_reward = self.discriminator.predict_reward

    def _safe_act_amp(alg_self, amp_obs: torch.Tensor) -> None:
      original_act_amp(_safe_nan_to_num(amp_obs))

    def _safe_process_amp_step(alg_self, amp_obs: torch.Tensor) -> None:
      original_process_amp_step(_safe_nan_to_num(amp_obs))

    def _safe_predict_reward(discriminator_self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
      row_valid = _finite_row_mask(state, next_state)
      self._rollout_safety_stats["amp_sanitized_envs"] += float((~row_valid).sum().item())
      safe_state = _safe_nan_to_num(state)
      safe_next_state = _safe_nan_to_num(next_state)
      rewards = original_predict_reward(safe_state, safe_next_state)
      rewards = rewards.reshape(-1)
      reward_finite = torch.isfinite(rewards)
      self._rollout_safety_stats["style_reward_sanitized_envs"] += float((~reward_finite).sum().item())
      rewards = _safe_nan_to_num(rewards)
      rewards = torch.where(row_valid, rewards, torch.zeros_like(rewards))
      rewards = torch.where(reward_finite, rewards, torch.zeros_like(rewards))
      return rewards

    self.alg.act_amp = types.MethodType(_safe_act_amp, self.alg)
    self.alg.process_amp_step = types.MethodType(_safe_process_amp_step, self.alg)
    self.discriminator.predict_reward = types.MethodType(_safe_predict_reward, self.discriminator)
    self.alg._z1_amp_safety_hooks_installed = True

  def _build_amp_safety_log(self) -> dict[str, torch.Tensor]:
    update_stats = getattr(self.alg, "last_safety_stats", {})
    return {
      "Diagnostics/amp_sanitized_envs": torch.tensor(
        self._rollout_safety_stats["amp_sanitized_envs"], device=self.device
      ),
      "Diagnostics/style_reward_sanitized_envs": torch.tensor(
        self._rollout_safety_stats["style_reward_sanitized_envs"], device=self.device
      ),
      "Diagnostics/amp_policy_samples_dropped": torch.tensor(
        float(update_stats.get("policy_samples_dropped", 0.0)), device=self.device
      ),
      "Diagnostics/amp_expert_samples_dropped": torch.tensor(
        float(update_stats.get("expert_samples_dropped", 0.0)), device=self.device
      ),
      "Diagnostics/amp_invalid_minibatches": torch.tensor(
        float(update_stats.get("invalid_minibatches", 0.0)), device=self.device
      ),
    }

  def _export_policy_to_onnx(self, path: str, filename: str = "policy.onnx"):
    module = self.alg.actor if RSL_RL_V4_PLUS else self.alg.actor_critic
    if not os.path.exists(path):
      os.makedirs(path, exist_ok=True)

    if hasattr(module, "as_onnx"):
      onnx_model = module.as_onnx(verbose=False)
      onnx_model.to("cpu")
      onnx_model.eval()
      torch.onnx.export(
        onnx_model,
        onnx_model.get_dummy_inputs(),  # type: ignore[attr-defined]
        os.path.join(path, filename),
        export_params=True,
        opset_version=18,
        verbose=False,
        input_names=onnx_model.input_names,  # type: ignore[attr-defined]
        output_names=onnx_model.output_names,  # type: ignore[attr-defined]
        dynamic_axes={},
        dynamo=False,
        **_onnx_export_kwargs_single_file(),
      )
      _inline_external_onnx_data(os.path.join(path, filename))
      return

    raise TypeError(f"Policy module {type(module)} does not support ONNX export via as_onnx().")

  def save(self, path: str, infos=None, **kwargs):
    save_onnx = kwargs.pop("save_onnx", False)
    super().save(path, infos, save_onnx=False, **kwargs)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    if save_onnx:
      self._export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    if save_onnx:
      onnx_path = os.path.join(policy_path, filename)
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      attach_metadata_to_onnx(onnx_path, metadata)
      _inline_external_onnx_data(onnx_path)
      if self.logger_type in ["wandb"]:
        wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

  def learn(self, *args, **kwargs):
    self._clamp_policy_std_()
    return super().learn(*args, **kwargs)

  def log(self, locs: dict, width: int = 80, pad: int = 35):
    locs = dict(locs)
    ep_infos = list(locs.get("ep_infos", []))
    diagnostics = self._build_amp_safety_log()
    if ep_infos:
      merged_first = dict(ep_infos[0])
      merged_first.update(diagnostics)
      ep_infos[0] = merged_first
    else:
      ep_infos = [diagnostics]
    locs["ep_infos"] = ep_infos
    try:
      return super().log(locs, width=width, pad=pad)
    finally:
      self._reset_amp_safety_stats()

  def load(self, *args, **kwargs):
    infos = super().load(*args, **kwargs)
    self._clamp_policy_std_()
    return infos
