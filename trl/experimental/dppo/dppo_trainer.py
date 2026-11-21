# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import textwrap

import torch

from ...extras.profiling import profiling_context
from ...trainer.grpo_trainer import GRPOTrainer
from ...trainer.utils import get_config_model_id, nanmax
from .dppo_config import DPPOConfig


# Probabilities are clamped away from 0 and 1 by this amount inside logs, as in the reference implementation
_EPS = 1e-8
_MASS_EPS = 1e-12


def binary_tv(logps: torch.Tensor, rollout_logps: torch.Tensor) -> torch.Tensor:
    r"""
    Binary approximation of the total variation between μ and π (Eq. 13 of the DPPO paper):
    $D^{\text{Bin}}_{\text{TV}} = |\mu(a_t) - \pi(a_t)|$.

    Args:
        logps (`torch.Tensor`): Log-probabilities of the sampled tokens under π, of any shape.
        rollout_logps (`torch.Tensor`): Log-probabilities of the same tokens under μ, same shape.
    """
    return (logps.float().exp() - rollout_logps.float().exp()).abs()


def binary_kl(logps: torch.Tensor, rollout_logps: torch.Tensor) -> torch.Tensor:
    r"""
    Binary approximation of $\mathrm{KL}(\mu \| \pi)$ (Eq. 14 of the DPPO paper), the KL divergence between the
    Bernoulli distributions {sampled token, every other token}:
    $\mu(a_t)\log\frac{\mu(a_t)}{\pi(a_t)} + (1-\mu(a_t))\log\frac{1-\mu(a_t)}{1-\pi(a_t)}$.
    """
    logps, rollout_logps = logps.float(), rollout_logps.float()
    p, mu = logps.exp(), rollout_logps.exp()
    kl = mu * (rollout_logps - logps) + (1.0 - mu) * torch.log((1.0 - mu + _EPS) / (1.0 - p + _EPS))
    return kl.clamp(min=0.0)  # non-negative up to rounding


def topk_divergences(
    score_logps: torch.Tensor, rollout_topk_logps: torch.Tensor, rollout_topk_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Top-K approximations of $\mathrm{KL}(\mu \| \pi)$ and of the total variation (Eqs. 15-16 of the DPPO paper).

    The support is $\mathcal{A}'' = \mathrm{TopK}(\mu, K) \cup \{a_t\} \cup \{\text{other}\}$, where "other" aggregates the
    mass of every remaining token. Both are lower bounds of the true divergences.

    Args:
        score_logps (`torch.Tensor` of shape `(..., K)`):
            Log-probabilities under π of the candidate tokens.
        rollout_topk_logps (`torch.Tensor` of shape `(..., K)`):
            Log-probabilities under μ of the same tokens. `-inf` (or NaN) marks an entry to ignore.
        rollout_topk_ids (`torch.Tensor` of shape `(..., K)`):
            Candidate token ids, padded with `-1`.

    Returns:
        Tuple `(kl, tv)` of tensors of shape `(...)`.
    """
    valid = (rollout_topk_ids >= 0) & torch.isfinite(rollout_topk_logps)
    mu_logps = torch.where(valid, rollout_topk_logps.float(), 0.0)
    pi_logps = torch.where(valid, score_logps.float(), 0.0)
    mu = torch.where(valid, mu_logps.exp(), 0.0)
    pi = torch.where(valid, pi_logps.exp(), 0.0)

    rest_mu = (1.0 - mu.sum(-1)).clamp(min=_MASS_EPS)
    rest_pi = (1.0 - pi.sum(-1)).clamp(min=_MASS_EPS)

    kl = (mu * (mu_logps - pi_logps)).sum(-1) + rest_mu * (rest_mu.log() - rest_pi.log())
    tv = 0.5 * ((mu - pi).abs().sum(-1) + (rest_mu - rest_pi).abs())
    return kl.clamp(min=0.0), tv


def dppo_mask(
    divergence: torch.Tensor,
    logps: torch.Tensor,
    rollout_logps: torch.Tensor,
    advantages: torch.Tensor,
    threshold_low: float,
    threshold_high: float,
) -> torch.Tensor:
    r"""
    DPPO trust-region mask (Eq. 12 of the DPPO paper). A token is masked (0) iff its update moves the probability ratio
    further from 1 *and* the divergence exceeds the threshold:

    - $\hat{A}_t > 0$, $\pi(a_t) > \mu(a_t)$ and $D > \delta_{\text{high}}$, or
    - $\hat{A}_t < 0$, $\pi(a_t) < \mu(a_t)$ and $D > \delta_{\text{low}}$.

    Returns a float tensor of 0s and 1s, broadcast from the inputs, with no gradient.
    """
    with torch.no_grad():
        increases = logps > rollout_logps
        decreases = logps < rollout_logps
        blocked_pos = (advantages > 0) & increases & (divergence > threshold_high)
        blocked_neg = (advantages < 0) & decreases & (divergence > threshold_low)
        return (~(blocked_pos | blocked_neg)).float()


class DPPOTrainer(GRPOTrainer):
    r"""
    Trainer for Divergence Proximal Policy Optimization (DPPO).

    DPPO ([Rethinking the Trust Region in LLM Reinforcement Learning](https://huggingface.co/papers/2602.04879))
    replaces PPO's heuristic ratio clipping, a noisy single-sample estimate of the policy change that over-penalizes
    low-probability tokens and under-constrains high-probability ones, with a mask based on a direct estimate of the
    divergence between the rollout policy μ and the current policy π (binary or top-K approximations of TV or KL). The
    surrogate is

    $$
    \mathcal{L}(\pi) = \mathbb{E}_{y \sim \mu}\Big[\sum_t M_t \cdot \min(r_t, C) \cdot \hat{A}_t\Big], \quad r_t =
    \frac{\pi(y_t|s_t)}{\mu(y_t|s_t)},
    $$

    where $M_t$ blocks a token only when its update moves $r_t$ away from 1 and the divergence exceeds δ, and the
    gradient flows through $\log \pi$ only (the clipped ratio is a detached weight).

    Generation, rewards and group-relative advantages are inherited from [`GRPOTrainer`]; only the loss and, for the
    `"topk_*"` divergences, the capture of the rollout's top-K log-probabilities change.

    Example:

    ```python
    from datasets import load_dataset

    from trl.experimental.dppo import DPPOConfig, DPPOTrainer
    from trl.rewards import accuracy_reward

    dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

    trainer = DPPOTrainer(
        model="Qwen/Qwen3-0.6B",
        reward_funcs=accuracy_reward,
        args=DPPOConfig(divergence_type="binary_tv", use_vllm=True),
        train_dataset=dataset,
    )
    trainer.train()
    ```
    """

    _tag_names = ["trl", "dppo"]
    _name = "DPPO"
    _paper = {
        "title": "Rethinking the Trust Region in LLM Reinforcement Learning",
        "id": "2602.04879",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @article{qi2026rethinking,
                title        = {{Rethinking the Trust Region in LLM Reinforcement Learning}},
                author       = {Penghui Qi and Xiangxin Zhou and Zichen Liu and Tianyu Pang and Chao Du and Min Lin and Wee Sun Lee},
                year         = 2026,
                journal      = {arXiv preprint arXiv:2602.04879}
            }"""),
    }

    def __init__(self, model, reward_funcs, args=None, **kwargs):
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            args = DPPOConfig(f"{model_name.split('/')[-1]}-DPPO")
        elif not isinstance(args, DPPOConfig):
            raise TypeError(f"`args` must be a `DPPOConfig`, got {type(args).__name__}.")

        super().__init__(model, reward_funcs, args=args, **kwargs)

        self.divergence_type = args.divergence_type
        self.divergence_threshold_low = args.divergence_threshold
        self.divergence_threshold_high = args.divergence_threshold_high
        self.max_importance_ratio = args.max_importance_ratio
        self.use_rollout_logprobs = args.use_rollout_logprobs
        self._use_topk = self.divergence_type.startswith("topk")
        self._rollout_topk = None  # top-K candidates of the latest single-turn vLLM generation

        if self._use_topk:
            if not self.use_vllm:
                raise ValueError(
                    f"`divergence_type={self.divergence_type!r}` needs the rollout's top-K log-probabilities, which "
                    "only vLLM returns. Set `use_vllm=True`, or use a `binary_*` divergence."
                )
            if self.tools or self.rollout_func is not None:
                raise NotImplementedError(
                    "The `topk_*` divergences don't support tool calling or `rollout_func` yet. Use a `binary_*` "
                    "divergence instead."
                )
            # Ask vLLM for the top-K logprobs of every generated token (the sampled token is always included too)
            self.vllm_generation.logprobs = args.divergence_top_k

    # ------------------------------------------------------------------------------------------------------------
    # Rollout: capture vLLM's top-K logprobs (topk_* divergences only)
    # ------------------------------------------------------------------------------------------------------------

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields, num_generations, has_tool_images=False):
        if not self._use_topk:
            return super()._generate_single_turn(
                prompt_ids, images, multimodal_fields, num_generations, has_tool_images=has_tool_images
            )

        if self.state.global_step != self._last_loaded_step:
            with profiling_context(self, "sync_weights"):
                self.vllm_generation.sync_weights()
            self._last_loaded_step = self.state.global_step

        _, completion_ids, topk_logprobs, topk_token_ids, sampling_mask = self.vllm_generation.generate(
            prompts=prompt_ids,
            images=images,
            num_generations=num_generations,
            profiler=profiling_context(self, "vLLM.generate"),
        )
        if topk_logprobs is None or topk_token_ids is None:
            raise RuntimeError("vLLM returned no top-K logprobs, which the `topk_*` divergences need.")

        # With logprobs=K, the candidates are sorted by rank, so the sampled token is not necessarily the first one:
        # look it up by id. vLLM always includes it, even outside the top-K.
        sampled_logprobs = []
        for seq_ids, seq_logprobs, seq_token_ids in zip(completion_ids, topk_logprobs, topk_token_ids, strict=True):
            seq_sampled = []
            for token_id, logprobs, token_ids in zip(seq_ids, seq_logprobs, seq_token_ids, strict=True):
                try:
                    seq_sampled.append(logprobs[token_ids.index(token_id)])
                except ValueError:
                    seq_sampled.append(None)  # mapped to NaN downstream: the token falls back to μ = π
            sampled_logprobs.append(seq_sampled)

        self._rollout_topk = (topk_token_ids, topk_logprobs)
        return completion_ids, sampled_logprobs, sampling_mask

    def _generate_and_score_completions(self, inputs):
        self._rollout_topk = None
        output = super()._generate_and_score_completions(inputs)

        if self._use_topk:
            topk_token_ids, topk_logprobs = self._rollout_topk
            self._rollout_topk = None
            num_sequences, seq_len = output["completion_ids"].shape
            num_candidates = max(
                (len(candidates) for seq in topk_token_ids for candidates in seq), default=1
            )  # K or K + 1 (when the sampled token falls outside the top-K)
            ids = torch.full((num_sequences, seq_len, num_candidates), -1, dtype=torch.long)
            logps = torch.full((num_sequences, seq_len, num_candidates), -math.inf, dtype=torch.float32)
            for i, (seq_ids, seq_logps) in enumerate(zip(topk_token_ids, topk_logprobs, strict=True)):
                for t, (cand_ids, cand_logps) in enumerate(zip(seq_ids, seq_logps, strict=True)):
                    n = len(cand_ids)
                    ids[i, t, :n] = torch.tensor(cand_ids, dtype=torch.long)
                    logps[i, t, :n] = torch.tensor(
                        [-math.inf if lp is None else lp for lp in cand_logps], dtype=torch.float32
                    )
            device = self.accelerator.device
            output["rollout_topk_ids"] = ids.to(device)
            output["rollout_topk_logps"] = logps.to(device)
        return output

    # ------------------------------------------------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------------------------------------------------

    def _compute_loss(self, model, inputs):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]

        outputs = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            compute_aux_loss=self.aux_loss_enabled,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            spatial_shapes=inputs.get("spatial_shapes"),
            num_tiles=inputs.get("num_tiles"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
            image_position_ids=inputs.get("image_position_ids"),
            sampling_mask=inputs.get("sampling_mask"),
            score_ids=inputs.get("rollout_topk_ids") if self._use_topk else None,
        )
        per_token_logps, entropies, aux_loss = outputs[:3]

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        advantages = inputs["advantages"]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)

        # Behavior policy μ: the vLLM rollout logprobs (paper default), else the trainer-recomputed logprobs before
        # the update, else (fully on-policy) π itself, in which case DPPO reduces to the policy gradient.
        logps_detached = per_token_logps.detach()
        rollout_logps = inputs.get("sampling_per_token_logps") if self.use_rollout_logprobs else None
        if rollout_logps is None:
            rollout_logps = inputs.get("old_per_token_logps")
        if rollout_logps is None:
            rollout_logps = logps_detached
        # vLLM can return a NaN logprob; such a token gets μ = π (ratio 1, zero divergence, never masked)
        rollout_logps = torch.where(torch.isfinite(rollout_logps), rollout_logps, logps_detached)

        # Divergence estimate D(μ || π) per token. It only gates the mask, so it carries no gradient.
        with torch.no_grad():
            if self.divergence_type == "binary_tv":
                divergence = binary_tv(logps_detached, rollout_logps)
            elif self.divergence_type == "binary_kl":
                divergence = binary_kl(logps_detached, rollout_logps)
            else:
                topk_kl, topk_tv = topk_divergences(
                    outputs[3].detach(), inputs["rollout_topk_logps"], inputs["rollout_topk_ids"]
                )
                divergence = topk_tv if self.divergence_type == "topk_tv" else topk_kl

        trust_region_mask = dppo_mask(
            divergence,
            logps_detached,
            rollout_logps,
            advantages,
            threshold_low=self.divergence_threshold_low,
            threshold_high=self.divergence_threshold_high,
        )

        # Importance ratio r_t = π / μ, used as a detached weight (optionally clipped at C). The surrogate
        # coef * (1 + log π - sg(log π)) has value coef and gradient coef * ∇log π, i.e. that of r_t when r_t <= C.
        log_ratio = torch.clamp(per_token_logps - rollout_logps, max=20.0)
        ratio = torch.exp(log_ratio.detach())
        coef = ratio if self.max_importance_ratio is None else ratio.clamp(max=self.max_importance_ratio)
        surrogate = coef * (1.0 + per_token_logps - logps_detached)
        per_token_loss = -trust_region_mask * surrogate * advantages

        if self.off_policy_mask_threshold is not None:
            off_policy_mask = self.get_off_policy_mask(
                advantages=advantages,
                per_token_logps=per_token_logps,
                sampling_per_token_logps=inputs.get("sampling_per_token_logps", rollout_logps),
                mask=mask,
                off_policy_threshold=self.off_policy_mask_threshold,
            )
            per_token_loss = per_token_loss * off_policy_mask

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            if self.args.use_bias_correction_kl:
                per_token_kl = per_token_kl * coef
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # Aggregation (same conventions as GRPOTrainer)
        mode = "train" if self.model.training else "eval"
        accumulation = self.current_gradient_accumulation_steps if mode == "train" else 1.0  # no accum in eval
        if self.loss_type == "grpo":
            loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean() / accumulation
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0) / accumulation
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / accumulation
        elif self.loss_type == "dapo":
            # `num_items_in_batch` spans the generation batch, so rescale it to one accumulation window
            normalizer = inputs["num_items_in_batch"].clamp(min=1.0) / self.accelerator.num_processes
            if mode == "train":
                normalizer = normalizer * self.current_gradient_accumulation_steps / self.args.steps_per_generation
            loss = (per_token_loss * mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type for DPPO: {self.loss_type}")

        if self.aux_loss_enabled:
            loss = loss + self.router_aux_loss_coef * aux_loss / accumulation
            self._metrics[mode]["aux_loss"].append(self.accelerator.gather_for_metrics(aux_loss).mean().item())

        # Metrics
        def global_masked_mean(x):
            local_sum, local_count = (x * mask).sum(), mask.sum().float()
            totals = self.accelerator.reduce(torch.stack([local_sum, local_count]), reduction="sum")
            return (totals[0] / totals[1].clamp(min=1.0)).item()

        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(global_masked_mean(per_token_kl))
        self._metrics[mode]["entropy"].append(global_masked_mean(entropies))

        blocked = 1.0 - trust_region_mask
        self._metrics[mode]["dppo/masked_ratio/high_mean"].append(global_masked_mean(blocked * (advantages > 0)))
        self._metrics[mode]["dppo/masked_ratio/low_mean"].append(global_masked_mean(blocked * (advantages < 0)))
        self._metrics[mode]["dppo/masked_ratio/region_mean"].append(global_masked_mean(blocked))
        self._metrics[mode]["dppo/divergence/mean"].append(global_masked_mean(divergence))
        divergence_max = (divergence * mask).max() if mask.any() else divergence.new_zeros(())
        self._metrics[mode]["dppo/divergence/max"].append(nanmax(self.accelerator.gather(divergence_max)).item())
        self._metrics[mode]["dppo/importance_ratio/mean"].append(global_masked_mean(ratio))
        if self.max_importance_ratio is not None:
            self._metrics[mode]["dppo/importance_ratio/clipped_mean"].append(
                global_masked_mean((ratio > self.max_importance_ratio).float())
            )

        return loss
