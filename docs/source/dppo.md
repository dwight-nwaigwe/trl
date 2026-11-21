# DPPO

In the paper [Rethinking the Trust Region in LLM Reinforcement Learning](https://huggingface.co/papers/2602.04879), the authors argue that PPO's ratio clipping is ill-suited to the large vocabularies of LLMs. The probability ratio of the sampled token is a noisy single-sample estimate of the policy change: it over-penalizes low-probability tokens (a rare token going from 0.1% to 0.3% has ratio 3 and is clipped) and under-constrains high-probability ones (a token going from 50% to 75% has ratio 1.5 but a large policy shift). Divergence Proximal Policy Optimization (DPPO) replaces the clip with a mask based on a direct estimate of the divergence between the rollout policy μ and the current policy π.

To use DPPO, you can use the [`experimental.dppo.DPPOTrainer`] class in `trl.experimental.dppo`.

## Objective

$$
\mathcal{L}^{\text{DPPO}}(\pi) = \mathbb{E}_{y \sim \mu} \left[ \sum_{t=1}^{|y|} M_t \cdot \min(r_t, C) \cdot \hat{A}_t \right], \qquad r_t = \frac{\pi(y_t|s_t)}{\mu(y_t|s_t)}
$$

where the mask blocks a token only when its update moves the ratio further from 1 *and* the divergence exceeds the threshold:

$$
M_t = \begin{cases} 0 & \text{if } (\hat{A}_t > 0,\ r_t > 1,\ D_t > \delta) \text{ or } (\hat{A}_t < 0,\ r_t < 1,\ D_t > \delta) \\ 1 & \text{otherwise} \end{cases}
$$

The clipped ratio  \\( \min(r_t, C) \\) is a detached weight, so the gradient is  \\( M_t \min(r_t, C) \hat{A}_t \nabla \log \pi(y_t|s_t) \\). Advantages are group-relative as in GRPO, without standard-deviation normalization (`scale_rewards="none"`).

μ is the rollout policy as scored by vLLM, so the trust region also bounds the training-inference mismatch. Without vLLM (or with `use_rollout_logprobs=False`), μ is the policy before the update as recomputed by the trainer; on the first update of a fully on-policy batch, μ = π and DPPO reduces to the policy gradient.

## Divergence approximations

| `divergence_type` | Divergence \\( D_t \\) | Default δ | Notes |
|---|---|---|---|
| `"binary_tv"` | \\( \lvert \mu(a_t) - \pi(a_t) \rvert \\) | 0.2 | Recommended; no overhead |
| `"binary_kl"` | KL between Bernoullis {a_t, rest} | 0.05 | No overhead |
| `"topk_tv"` | TV over TopK(μ) ∪ {a_t} ∪ {other} | 0.2 | Requires vLLM |
| `"topk_kl"` | KL over TopK(μ) ∪ {a_t} ∪ {other} | 0.05 | Requires vLLM |

The top-K variants ask vLLM for the top `divergence_top_k` (default 20, vLLM's default `max_logprobs`) log-probabilities of every generated token, and score those same tokens under π with the fused LM head, without materializing the logits. They don't support tool calling or `rollout_func` yet.

## Usage

```python
from trl.experimental.dppo import DPPOConfig, DPPOTrainer

training_args = DPPOConfig(
    divergence_type="binary_tv",
    divergence_threshold=0.2,  # δ; the paper uses 0.15 with LoRA
    max_importance_ratio=5.0,  # C
    use_vllm=True,
    num_generations=16,
    temperature=1.0,
    learning_rate=1e-6,
)
trainer = DPPOTrainer(
    model="Qwen/Qwen3-0.6B",
    reward_funcs=...,
    train_dataset=...,
    args=training_args,
)
trainer.train()
```

Because DPPO doesn't clip the ratio, `epsilon`, `epsilon_high` and `delta` are ignored, and `loss_type` only selects how per-token losses are aggregated (`"dapo"`, the default, is a token mean over the batch). `vllm_importance_sampling_correction` must stay `False`: the DPPO ratio is already taken w.r.t. the rollout policy.

Logged metrics: `dppo/masked_ratio/{low,high,region}_mean` (fraction of tokens blocked by the trust region, for negative/positive/all advantages), `dppo/divergence/{mean,max}` and `dppo/importance_ratio/{mean,clipped_mean}`.

## DPPOTrainer

[[autodoc]] experimental.dppo.DPPOTrainer
    - train
    - save_model
    - push_to_hub

## DPPOConfig

[[autodoc]] experimental.dppo.DPPOConfig
