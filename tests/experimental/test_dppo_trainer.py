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

import pytest
import torch
from datasets import load_dataset

from trl.experimental.dppo import DPPOConfig, DPPOTrainer, binary_kl, binary_tv, dppo_mask, topk_divergences

from ..testing_utils import TrlTestCase


def _log(p):
    return torch.log(torch.tensor(p, dtype=torch.float32))


class TestDivergences:
    def test_binary_tv(self):
        logps, rollout_logps = _log([0.9, 0.1, 0.5]), _log([0.6, 0.3, 0.5])
        torch.testing.assert_close(binary_tv(logps, rollout_logps), torch.tensor([0.3, 0.2, 0.0]))

    def test_binary_kl_matches_closed_form(self):
        p, mu = 0.7, 0.4
        expected = mu * math.log(mu / p) + (1 - mu) * math.log((1 - mu) / (1 - p))
        torch.testing.assert_close(binary_kl(_log([p]), _log([mu])), torch.tensor([expected]), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(binary_kl(_log([0.3]), _log([0.3])), torch.tensor([0.0]), atol=1e-6, rtol=0)

    def test_topk_divergences_match_exact_when_support_covers_vocab(self):
        # With every token of the vocabulary in the support, the "other" bucket is empty and the approximations are
        # the exact divergences
        torch.manual_seed(0)
        mu_logits, pi_logits = torch.randn(4, 6), torch.randn(4, 6)
        mu_logps, pi_logps = mu_logits.log_softmax(-1), pi_logits.log_softmax(-1)
        ids = torch.arange(6).expand(4, 6)
        kl, tv = topk_divergences(pi_logps, mu_logps, ids)
        exact_kl = (mu_logps.exp() * (mu_logps - pi_logps)).sum(-1)
        exact_tv = 0.5 * (mu_logps.exp() - pi_logps.exp()).abs().sum(-1)
        torch.testing.assert_close(kl, exact_kl, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(tv, exact_tv, atol=1e-5, rtol=1e-4)

    def test_topk_divergences_are_lower_bounds_and_ignore_padding(self):
        torch.manual_seed(0)
        mu_logps, pi_logps = torch.randn(8, 50).log_softmax(-1), torch.randn(8, 50).log_softmax(-1)
        k = 5
        ids = mu_logps.topk(k, dim=-1).indices
        # Pad one extra slot with -1 to check it is ignored
        ids_padded = torch.cat([ids, torch.full((8, 1), -1)], dim=-1)
        mu_topk = torch.cat([mu_logps.gather(-1, ids), torch.full((8, 1), -math.inf)], dim=-1)
        pi_topk = torch.cat([pi_logps.gather(-1, ids), torch.zeros(8, 1)], dim=-1)  # the kernel returns 0 at padding
        kl, tv = topk_divergences(pi_topk, mu_topk, ids_padded)
        kl_ref, tv_ref = topk_divergences(pi_logps.gather(-1, ids), mu_logps.gather(-1, ids), ids)
        torch.testing.assert_close(kl, kl_ref)
        torch.testing.assert_close(tv, tv_ref)
        exact_kl = (mu_logps.exp() * (mu_logps - pi_logps)).sum(-1)
        exact_tv = 0.5 * (mu_logps.exp() - pi_logps.exp()).abs().sum(-1)
        assert torch.all(kl <= exact_kl + 1e-5)
        assert torch.all(tv <= exact_tv + 1e-5)
        # The binary partition {a, rest} is coarser than the top-K one when a is in the support, so its TV is smaller
        top1 = ids[:, :1]
        assert torch.all(binary_tv(pi_logps.gather(-1, top1), mu_logps.gather(-1, top1)).squeeze(-1) <= tv + 1e-6)


class TestDPPOMask:
    def test_mask_blocks_only_updates_leaving_the_trust_region(self):
        #                 A>0,π↑,D>δ  A>0,π↓,D>δ  A<0,π↓,D>δ  A<0,π↑,D>δ  A>0,π↑,D<δ  A=0,π↑,D>δ
        advantages = torch.tensor([1.0, 1.0, -1.0, -1.0, 1.0, 0.0])
        rollout_logps = _log([0.5] * 6)
        logps = _log([0.9, 0.1, 0.1, 0.9, 0.6, 0.9])
        divergence = binary_tv(logps, rollout_logps)
        mask = dppo_mask(divergence, logps, rollout_logps, advantages, threshold_low=0.2, threshold_high=0.2)
        torch.testing.assert_close(mask, torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 1.0]))

    def test_asymmetric_thresholds(self):
        advantages = torch.tensor([1.0, -1.0])
        rollout_logps = _log([0.5, 0.5])
        logps = _log([0.8, 0.2])  # TV = 0.3 for both
        divergence = binary_tv(logps, rollout_logps)
        mask = dppo_mask(divergence, logps, rollout_logps, advantages, threshold_low=0.25, threshold_high=0.35)
        torch.testing.assert_close(mask, torch.tensor([1.0, 0.0]))

    def test_mask_is_wider_than_ppo_for_rare_tokens(self):
        # A rare token whose probability triples has ratio 3 (clipped by PPO with ε=0.2), but a tiny TV: DPPO keeps it
        logps, rollout_logps = _log([0.003]), _log([0.001])
        divergence = binary_tv(logps, rollout_logps)
        mask = dppo_mask(divergence, logps, rollout_logps, torch.tensor([1.0]), threshold_low=0.2, threshold_high=0.2)
        assert mask.item() == 1.0
        # A frequent token whose probability moves 0.5 -> 0.75 has ratio 1.5 but TV 0.25 > 0.2: DPPO blocks it
        logps, rollout_logps = _log([0.75]), _log([0.5])
        divergence = binary_tv(logps, rollout_logps)
        mask = dppo_mask(divergence, logps, rollout_logps, torch.tensor([1.0]), threshold_low=0.2, threshold_high=0.2)
        assert mask.item() == 0.0


class TestDPPOConfig:
    def test_defaults(self):
        args = DPPOConfig("dummy")
        assert args.divergence_type == "binary_tv"
        assert args.divergence_threshold == 0.2
        assert args.divergence_threshold_high == 0.2
        assert args.max_importance_ratio == 5.0
        assert args.scale_rewards == "none"
        assert args.vllm_importance_sampling_correction is False

    def test_kl_default_threshold(self):
        args = DPPOConfig("dummy", divergence_type="binary_kl")
        assert args.divergence_threshold == 0.05

    def test_threshold_high_defaults_to_threshold(self):
        args = DPPOConfig("dummy", divergence_threshold=0.1)
        assert args.divergence_threshold_high == 0.1

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"divergence_type": "js"},
            {"loss_type": "cispo"},
            {"importance_sampling_level": "sequence"},
            {"vllm_importance_sampling_correction": True},
            {"divergence_type": "topk_tv", "use_rollout_logprobs": False},
            {"entropy_coef": 0.01},
            {"max_importance_ratio": 0.0},
        ],
    )
    def test_invalid(self, kwargs):
        with pytest.raises(ValueError):
            DPPOConfig("dummy", **kwargs)


class TestDPPOTrainer(TrlTestCase):
    @pytest.mark.parametrize("divergence_type", ["binary_tv", "binary_kl"])
    @pytest.mark.parametrize("loss_type", ["dapo", "grpo"])
    def test_train(self, divergence_type, loss_type):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = DPPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            num_iterations=2,  # off-policy updates, so the trust region is actually exercised
            divergence_type=divergence_type,
            loss_type=loss_type,
            report_to="none",
        )
        trainer = DPPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        logged = {k for log in trainer.state.log_history for k in log}
        assert "dppo/masked_ratio/region_mean" in logged
        assert "dppo/divergence/mean" in logged

        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_topk_requires_vllm(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        training_args = DPPOConfig(output_dir=self.tmp_dir, divergence_type="topk_tv", report_to="none")
        with pytest.raises(ValueError, match="vLLM"):
            DPPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

    def test_loss_gradient_matches_masked_ratio_objective(self):
        # The surrogate used in the loss must have the gradient of -M * min(r, C) * A, with M and C detached
        torch.manual_seed(0)
        logps = torch.randn(2, 5).log_softmax(-1).requires_grad_()
        rollout_logps = torch.randn(2, 5).log_softmax(-1)
        advantages = torch.tensor([[1.0], [-1.0]])
        mask = dppo_mask(
            binary_tv(logps.detach(), rollout_logps),
            logps.detach(),
            rollout_logps,
            advantages,
            threshold_low=0.05,
            threshold_high=0.05,
        )
        cap = 1.5
        ratio = torch.exp(logps.detach() - rollout_logps)
        coef = ratio.clamp(max=cap)
        surrogate = coef * (1.0 + logps - logps.detach())
        (-(mask * surrogate * advantages).sum()).backward()
        expected = -(mask * coef * advantages)  # d/dlogπ of -M * r * A where r <= C, -M * C * A elsewhere
        torch.testing.assert_close(logps.grad, expected)
