"""
SAC + HER training with a KL penalty against a frozen reference policy.

Implements Mitigation 2 from mitigation.md:
    r' = r_proxy - β * (log π_curr(a|s) - log π_ref(a|s))

The per-step log-ratio is an unbiased estimator of KL(π_curr || π_ref) when
actions are sampled from π_curr — the standard RLHF formulation, applied
here to off-policy SAC. We override `SAC.train` and recompute the KL term
on each sampled minibatch so the penalty always reflects the *current*
policy's drift from the (frozen) reference, even as transitions sit in the
replay buffer.

Requires a pretrained reference policy with the same observation/action
spaces — see `scripts/train_reference.py`.
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Optional

import gymnasium as gym
import gymnasium_robotics  # noqa: F401 — registers Fetch envs
import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import HerReplayBuffer, SAC
from stable_baselines3.common.utils import polyak_update

from safety import SafetyMetricWrapper, evaluate_with_safety
from .training import HER_KWARGS, SAC_KWARGS, _make_train_env


def compute_action_log_prob(sac_model: SAC, obs, actions: th.Tensor) -> th.Tensor:
    """Compute log π(a|s) under `sac_model.policy.actor` for a given (obs, action) batch.

    The SAC actor uses a squashed (tanh) diagonal Gaussian. We pull the
    distribution params (mean, log_std) from the actor and then evaluate
    `log_prob(action)` using the `SquashedDiagGaussianDistribution` —
    which handles the inverse-tanh + Jacobian correction internally.

    Returns: tensor of shape (batch_size,).
    """
    actor = sac_model.policy.actor
    mean, log_std, _ = actor.get_action_dist_params(obs)
    return actor.action_dist.proba_distribution(mean, log_std).log_prob(actions)


class KLConstrainedSAC(SAC):
    """SAC with a per-step KL penalty against a frozen reference policy.

    `reference_model` must be a pretrained SAC with matching obs/action spaces.
    `kl_beta` is the penalty coefficient; β=0 disables the penalty.

    NOTE: this overrides `SAC.train()` for SB3 2.8.x. If you upgrade SB3 and
    the upstream `train()` signature drifts, this override needs updating.
    """

    def __init__(
        self,
        *args,
        reference_model: Optional[SAC] = None,
        kl_beta: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reference_model = reference_model
        self.kl_beta = kl_beta

        if self.reference_model is not None:
            # Freeze reference parameters and put it in eval mode.
            for p in self.reference_model.policy.parameters():
                p.requires_grad_(False)
            self.reference_model.policy.set_training_mode(False)
            # Match devices so log_prob can be computed without copies.
            self.reference_model.policy.to(self.device)

    # The body below is a copy of SB3 2.8.0 `SAC.train` with two added blocks:
    # one to compute the per-step KL log-ratio, and one to subtract β·KL from
    # the rewards used in the Bellman target.
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        kl_step_means: list[float] = []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            # ── KL penalty: r' = r - β · clip(log π_curr(a|s) - log π_ref(a|s)) ──
            # Clipping is critical: the reference policy is near-deterministic
            # (trained SAC converges to very low std), so log π_ref(a|s) can be
            # astronomically negative for any off-mode action, making the raw
            # KL term blow up to 10^3-10^5. We clip per-step KL to a sane range
            # so the penalty stays informative without overwhelming the proxy reward.
            KL_CLIP_MAX = 20.0  # ~e^20 ≈ 5e8 likelihood ratio — generous ceiling
            rewards = replay_data.rewards
            if self.reference_model is not None and self.kl_beta > 0.0:
                with th.no_grad():
                    log_p_curr = compute_action_log_prob(
                        self, replay_data.observations, replay_data.actions
                    )
                    log_p_ref = compute_action_log_prob(
                        self.reference_model, replay_data.observations, replay_data.actions
                    )
                    kl_step = (log_p_curr - log_p_ref).reshape(-1, 1)
                    kl_step = th.clamp(kl_step, min=-KL_CLIP_MAX, max=KL_CLIP_MAX)
                    rewards = rewards - self.kl_beta * kl_step
                    kl_step_means.append(float(kl_step.mean().item()))
            # ─────────────────────────────────────────────────────────────────

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        if len(kl_step_means) > 0:
            self.logger.record("train/kl_step_mean", float(np.mean(kl_step_means)))
            self.logger.record("train/kl_beta", self.kl_beta)


def _make_kl_model(
    env: gym.Env,
    reference_model: SAC,
    kl_beta: float,
    seed: int = 42,
    tensorboard_log: Optional[str] = None,
    verbose: int = 0,
) -> KLConstrainedSAC:
    """Construct a KLConstrainedSAC with the project-standard hyperparameters."""
    return KLConstrainedSAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=HER_KWARGS,
        verbose=verbose,
        seed=seed,
        tensorboard_log=tensorboard_log,
        reference_model=reference_model,
        kl_beta=kl_beta,
        **SAC_KWARGS,
    )


def train_with_kl_checkpoints(
    env_id: str,
    reward_fn: Callable,
    reference_model: SAC,
    kl_beta: float,
    timesteps: int,
    checkpoints: list[int],
    seed: int = 42,
    eval_episodes: int = 100,
    save_dir: str = "models/kl",
    results_dir: str = "results/kl",
    label: str = "",
    tensorboard_log: Optional[str] = None,
    verbose: int = 1,
) -> dict[int, dict]:
    """Train KLConstrainedSAC + HER, saving + evaluating at each checkpoint.

    Mirrors `train_with_checkpoints` but uses the KL-modified SAC and records
    `kl_beta` in each result JSON.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    checkpoints = sorted(set(checkpoints))
    if timesteps not in checkpoints:
        checkpoints.append(timesteps)

    train_env = _make_train_env(env_id, reward_fn)
    model = _make_kl_model(
        train_env,
        reference_model,
        kl_beta,
        seed=seed,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
    )

    all_metrics: dict[int, dict] = {}
    steps_trained = 0
    start_time = time.time()

    for checkpoint in checkpoints:
        steps_to_train = checkpoint - steps_trained
        if steps_to_train <= 0:
            continue

        print(f"\n  Training {steps_trained//1000}k → {checkpoint//1000}k ({steps_to_train:,} steps)...")
        model.learn(total_timesteps=steps_to_train, reset_num_timesteps=False)
        steps_trained = checkpoint

        model_path = os.path.join(save_dir, f"{label}_{checkpoint//1000}k")
        model.save(model_path)
        # Ensure .zip extension for consistency with eval_local.py expectations
        import glob
        saved_file = model_path if os.path.exists(model_path) else model_path + ".zip"
        if os.path.exists(model_path) and not model_path.endswith(".zip"):
            os.rename(model_path, model_path + ".zip")
            saved_file = model_path + ".zip"
        print(f"  Saved: {saved_file}")

        print(f"  Evaluating ({eval_episodes} episodes)...")
        eval_env = SafetyMetricWrapper(gym.make(env_id))
        metrics = evaluate_with_safety(model, eval_env, n_episodes=eval_episodes, reward_fn=reward_fn)
        eval_env.close()

        metrics["checkpoint_steps"] = checkpoint
        metrics["kl_beta"] = kl_beta
        results_path = os.path.join(results_dir, f"{label}_{checkpoint//1000}k.json")
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(
            f"  {checkpoint//1000}k: success={metrics['success_rate']:.1%} | "
            f"hacking={metrics.get('hacking_rate', 'N/A')} | "
            f"action_mag={metrics['mean_action_mag']:.4f}"
        )
        all_metrics[checkpoint] = metrics

    train_time = time.time() - start_time
    train_env.close()

    summary = {
        "config": {
            "env_id": env_id,
            "label": label,
            "checkpoints": checkpoints,
            "eval_episodes": eval_episodes,
            "seed": seed,
            "kl_beta": kl_beta,
            "train_time_seconds": train_time,
        },
        "results_by_checkpoint": {
            f"{c//1000}k": all_metrics[c] for c in checkpoints if c in all_metrics
        },
    }
    summary_path = os.path.join(results_dir, f"{label}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Total train time: {train_time/60:.1f} min")
    print(f"  Summary: {summary_path}")

    return all_metrics
