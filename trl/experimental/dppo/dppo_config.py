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

from dataclasses import dataclass, field

from ...trainer.grpo_config import GRPOConfig


DIVERGENCE_TYPES = ("binary_tv", "binary_kl", "topk_tv", "topk_kl")

# Default trust-region radius per divergence, from the scaling experiments of the paper (App. E, F.3)
DEFAULT_DIVERGENCE_THRESHOLDS = {"binary_tv": 0.2, "binary_kl": 0.05, "topk_tv": 0.2, "topk_kl": 0.05}


@dataclass
class DPPOConfig(GRPOConfig):
    r"""
    Configuration class for the [`experimental.dppo.DPPOTrainer`].

    [`experimental.dppo.DPPOConfig`] inherits every parameter from [`GRPOConfig`] and adds the parameters of
    Divergence Proximal Policy Optimization (DPPO), introduced in [Rethinking the Trust Region in LLM Reinforcement
    Learning](https://huggingface.co/papers/2602.04879). DPPO replaces PPO's ratio clipping with a mask based on a
    direct estimate of the divergence between the rollout policy μ and the current policy π: a token's update is
    blocked only if it moves the ratio further away from 1 *and* the divergence exceeds a threshold δ.

    Because DPPO does not clip the probability ratio, `epsilon`, `epsilon_high` and `delta` are ignored. `loss_type`
    only selects how the per-token losses are aggregated (`"dapo"`, `"bnpo"`, `"grpo"` or `"dr_grpo"`).

    Parameters:
        divergence_type (`str`, *optional*, defaults to `"binary_tv"`):
            Divergence approximation used for the trust region. One of:

            - `"binary_tv"`: total variation between the Bernoulli distributions {sampled token, rest}, i.e.
              `|μ(a_t) - π(a_t)|` (Eq. 13). Recommended by the paper; no extra memory or compute.
            - `"binary_kl"`: KL divergence `KL(μ || π)` between the same Bernoulli distributions (Eq. 14).
            - `"topk_tv"`: total variation over the rollout's top-K tokens, the sampled token and an "other" bucket
              (Eq. 15). Requires vLLM.
            - `"topk_kl"`: KL divergence `KL(μ || π)` over the same support (Eq. 16). Requires vLLM.

        divergence_threshold (`float`, *optional*):
            Trust-region radius δ. Applied to tokens with a negative advantage, and to tokens with a positive advantage
            unless `divergence_threshold_high` is set. Defaults to `0.2` for the TV variants and `0.05` for the KL
            variants, the values of the paper's scaling experiments.
        divergence_threshold_high (`float`, *optional*):
            Trust-region radius for tokens with a positive advantage. Defaults to `divergence_threshold`.
        divergence_top_k (`int`, *optional*, defaults to `20`):
            Number K of top tokens of the rollout distribution used by the `"topk_*"` divergences. vLLM returns at most
            `max_logprobs` (20 by default) candidates per token.
        max_importance_ratio (`float` or `None`, *optional*, defaults to `5.0`):
            Upper bound C on the (detached) importance ratio `π / μ` that weights each token's gradient. The paper uses
            `C = 5` in its scaling experiments. `None` disables the bound.
        use_rollout_logprobs (`bool`, *optional*, defaults to `True`):
            Whether μ is the rollout policy as scored by vLLM (as in the paper), so that the trust region also bounds
            the training-inference mismatch. If `False`, or when vLLM is not used, μ is the policy before the update
            as recomputed by the trainer (the paper's "recompute" ablation). Must be `True` for the `"topk_*"`
            divergences.

        > Parameters that control the default values of inherited parameters

        scale_rewards (`str`, *optional*, defaults to `"none"`):
            The paper uses group-relative advantages without standard-deviation normalization.
        vllm_importance_sampling_correction (`bool`, *optional*, defaults to `False`):
            Must be `False`: DPPO's ratio is already taken w.r.t. the rollout policy, so the extra truncated importance
            sampling correction of [`GRPOTrainer`] would apply the training-inference correction twice.
    """

    divergence_type: str = field(
        default="binary_tv",
        metadata={
            "help": "Divergence approximation used for the trust region.",
            "choices": list(DIVERGENCE_TYPES),
        },
    )
    divergence_threshold: float | None = field(
        default=None,
        metadata={
            "help": "Trust-region radius δ. Defaults to 0.2 for the TV variants and 0.05 for the KL variants. Applied "
            "to positive-advantage tokens too unless `divergence_threshold_high` is set."
        },
    )
    divergence_threshold_high: float | None = field(
        default=None,
        metadata={
            "help": "Trust-region radius for tokens with a positive advantage. Defaults to `divergence_threshold`."
        },
    )
    divergence_top_k: int = field(
        default=20,
        metadata={"help": "Number of top rollout tokens used by the `topk_*` divergences."},
    )
    max_importance_ratio: float | None = field(
        default=5.0,
        metadata={
            "help": "Upper bound C on the detached importance ratio weighting each token's gradient. `None` disables "
            "it."
        },
    )
    use_rollout_logprobs: bool = field(
        default=True,
        metadata={
            "help": "Whether μ is the vLLM rollout policy (paper default). If False, or without vLLM, μ is the "
            "trainer-recomputed policy before the update."
        },
    )

    # Parameters whose default values are overridden from GRPOConfig
    scale_rewards: str = field(
        default="none",
        metadata={
            "help": "Specifies the scaling strategy for rewards. DPPO uses group-relative advantages without "
            "standard-deviation normalization.",
        },
    )
    vllm_importance_sampling_correction: bool = field(
        default=False,
        metadata={
            "help": "Must be False: DPPO's ratio is already taken w.r.t. the rollout policy, so the correction would "
            "be applied twice."
        },
    )

    def __post_init__(self):
        super().__post_init__()

        if self.divergence_type not in DIVERGENCE_TYPES:
            raise ValueError(
                f"`divergence_type` must be one of {list(DIVERGENCE_TYPES)}, got {self.divergence_type!r}."
            )
        if self.divergence_threshold is None:
            self.divergence_threshold = DEFAULT_DIVERGENCE_THRESHOLDS[self.divergence_type]
        if self.divergence_threshold_high is None:
            self.divergence_threshold_high = self.divergence_threshold
        if self.divergence_threshold < 0 or self.divergence_threshold_high < 0:
            raise ValueError("`divergence_threshold` and `divergence_threshold_high` must be non-negative.")
        if self.divergence_top_k < 1:
            raise ValueError(f"`divergence_top_k` must be at least 1, got {self.divergence_top_k}.")
        if self.max_importance_ratio is not None and self.max_importance_ratio <= 0:
            raise ValueError(f"`max_importance_ratio` must be positive or None, got {self.max_importance_ratio}.")
        if self.loss_type not in ("dapo", "bnpo", "grpo", "dr_grpo"):
            raise ValueError(
                "In DPPO, `loss_type` only selects the loss aggregation and must be one of 'dapo', 'bnpo', 'grpo' or "
                f"'dr_grpo', got {self.loss_type!r}."
            )
        if self.importance_sampling_level != "token":
            raise ValueError("DPPO is a token-level method: `importance_sampling_level` must be 'token'.")
        if self.vllm_importance_sampling_correction:
            raise ValueError(
                "DPPO already takes its importance ratio and trust region w.r.t. the vLLM rollout policy, so "
                "`vllm_importance_sampling_correction` must be False (it would apply the correction twice)."
            )
        if self.divergence_type.startswith("topk") and not self.use_rollout_logprobs:
            raise ValueError(
                "The `topk_*` divergences need the rollout distribution: set `use_rollout_logprobs=True`."
            )
        if self.entropy_coef != 0.0 or self.use_adaptive_entropy:
            raise ValueError("The entropy bonus (`entropy_coef`, `use_adaptive_entropy`) is not supported by DPPO.")
