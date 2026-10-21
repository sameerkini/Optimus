"""Run: .venv/bin/python -m unittest discover -s tests -v"""

import math
import unittest

import torch
from torch.nn import functional as F

from src.attention import CausalSelfAttention, causal_mask
from src.cache import KVCache
from src.config import ModelConfig
from src.layers import RotaryPositionalEmbedding


def reference_rope(x, start_pos=0, base=10000.0):
    """Independent reference: explicitly multiply each 2D rotation matrix."""
    d_k, n = x.shape[-2:]
    columns = []
    for i in range(n):
        pairs = []
        for r in range(d_k // 2):
            angle = (start_pos + i) / base ** (2 * r / d_k)
            rotation = x.new_tensor([[math.cos(angle), -math.sin(angle)],
                                     [math.sin(angle), math.cos(angle)]])
            pairs.append((rotation @ x[..., 2*r:2*r+2, i:i+1]).squeeze(-1))
        columns.append(torch.cat(pairs, dim=-1))
    return torch.stack(columns, dim=-1)


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_mask_orientation_and_cached_offset(self):
        expected = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1],
                                 [0, 0, 1, 1], [0, 0, 0, 1]], dtype=torch.bool)
        self.assertTrue(torch.equal(causal_mask(4), expected))
        rectangular = torch.tensor([[1, 1], [1, 1], [1, 1], [1, 1], [0, 1]], dtype=torch.bool)
        self.assertTrue(torch.equal(causal_mask(2, start_pos=3), rectangular))
        self.assertTrue(causal_mask(1, start_pos=6).all())

    def test_rope_matches_rotation_matrices(self):
        rope = RotaryPositionalEmbedding(4).double()
        x = torch.randn(2, 3, 4, 5, dtype=torch.double)
        torch.testing.assert_close(rope(x, start_pos=7), reference_rope(x, 7), atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(rope(x)[..., :1], x[..., :1])
        self.assertEqual(list(rope.parameters()), [])

    def test_rope_preserves_lengths_and_relative_dot_products(self):
        rope = RotaryPositionalEmbedding(4).double()
        q, k = torch.randn(2, 4, 1, dtype=torch.double)
        torch.testing.assert_close(rope(q, 12).square().sum(), q.square().sum())
        before = rope(k, 3).T @ rope(q, 9)
        after = rope(k, 103).T @ rope(q, 109)
        torch.testing.assert_close(before, after, atol=1e-12, rtol=1e-12)
        x = torch.randn(2, 4, 6, dtype=torch.double)
        torch.testing.assert_close(rope(x)[..., 4:], rope(x[..., 4:], start_pos=4))

    def test_forward_and_gradients_match_independent_attention(self):
        model = CausalSelfAttention(8, 2, layer_idx=0).double()
        x = torch.randn(2, 8, 5, dtype=torch.double, requires_grad=True)
        actual, probabilities = model(x, return_attention=True)
        head_results = []
        for head in model.heads:
            def project(linear):
                return F.linear(x.transpose(-2, -1), linear.weight, linear.bias[:, 0]).transpose(-2, -1)
            q = reference_rope(project(head.query))
            k = reference_rope(project(head.key))
            v = project(head.value_down)
            mixed = F.scaled_dot_product_attention(q.transpose(-2, -1), k.transpose(-2, -1),
                                                   v.transpose(-2, -1), is_causal=True)
            head_results.append(F.linear(mixed, head.value_up.weight,
                                         head.value_up.bias[:, 0]).transpose(-2, -1))
        expected = torch.stack(head_results).sum(dim=0)
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-7)
        parameters = (x, *model.parameters())
        actual_grad = torch.autograd.grad(actual.square().mean(), parameters)
        expected_grad = torch.autograd.grad(expected.square().mean(), parameters)
        for left, right in zip(actual_grad, expected_grad):
            self.assertTrue(torch.isfinite(left).all())
            torch.testing.assert_close(left, right, atol=1e-9, rtol=1e-6)
        torch.testing.assert_close(probabilities.sum(dim=-2), torch.ones(2, 2, 5, dtype=torch.double))
        self.assertTrue((probabilities >= 0).all())
        self.assertTrue((probabilities.masked_select(~causal_mask(5)) == 0).all())

    def test_uniform_scores_produce_prefix_means(self):
        model = CausalSelfAttention(4, 1, layer_idx=0).double()
        head = model.heads[0]
        with torch.no_grad():
            for linear in (head.query, head.key):
                linear.weight.zero_(); linear.bias.zero_()
            for linear in (head.value_down, head.value_up):
                linear.weight.copy_(torch.eye(4)); linear.bias.zero_()
        x = torch.arange(20, dtype=torch.double).reshape(1, 4, 5)
        expected = x.cumsum(dim=-1) / torch.arange(1, 6, dtype=torch.double)
        torch.testing.assert_close(model(x), expected)

    def test_future_and_other_examples_cannot_change_prefix(self):
        model = CausalSelfAttention(8, 2, layer_idx=0).double()
        x = torch.randn(2, 8, 6, dtype=torch.double)
        changed = x.clone()
        changed[0, :, 3:] += 100
        original = model(x)
        updated = model(changed)
        torch.testing.assert_close(original[0, :, :3], updated[0, :, :3], atol=0, rtol=0)
        torch.testing.assert_close(original[1], updated[1], atol=0, rtol=0)

    def test_cached_single_tokens_match_full_pass(self):
        model = CausalSelfAttention(8, 2, layer_idx=3).double().eval()
        x = torch.randn(2, 8, 7, dtype=torch.double)
        cache = KVCache(n_max=7)
        projected_lengths = []
        hook = model.heads[0].key.register_forward_pre_hook(
            lambda module, args: projected_lengths.append(args[0].shape[-1]))
        with torch.no_grad():
            full = model(x)
            projected_lengths.clear()
            outputs = [model(x[..., :3], cache=cache)]
            for i in range(3, 7):
                outputs.append(model(x[..., i:i+1], cache=cache))
            cached = torch.cat(outputs, dim=-1)
            torch.testing.assert_close(full, cached, atol=1e-12, rtol=1e-12)
            raw_keys = torch.stack([head.key(x) for head in model.heads], dim=1)
            raw_values = torch.stack([head.value_down(x) for head in model.heads], dim=1)
            torch.testing.assert_close(cache.get(3).keys, model.rope(raw_keys))
            torch.testing.assert_close(cache.get(3).values, raw_values)
        hook.remove()
        self.assertEqual(projected_lengths[:5], [3, 1, 1, 1, 1])
        self.assertEqual(cache.length(3), 7)
        self.assertFalse(cache.get(3).keys.requires_grad)

    def test_shared_cache_two_layers_and_multi_token_chunks(self):
        layers = [CausalSelfAttention(8, 2, layer_idx=i).double().eval() for i in range(2)]
        x = torch.randn(2, 8, 7, dtype=torch.double)
        cache = KVCache(n_max=7)
        def run(value, cached=None):
            for layer in layers:
                value = value + layer(value, cache=cached)
            return value
        with torch.no_grad():
            full = run(x)
            pieces = [run(x[..., :3], cache), run(x[..., 3:5], cache), run(x[..., 5:], cache)]
        torch.testing.assert_close(torch.cat(pieces, dim=-1), full, atol=1e-12, rtol=1e-12)
        self.assertEqual(cache.length(0), 7)
        self.assertEqual(cache.length(1), 7)
        self.assertNotEqual(cache.get(0).keys.data_ptr(), cache.get(1).keys.data_ptr())

    def test_custom_masks_restrict_but_never_remove_causality(self):
        model = CausalSelfAttention(8, 2, layer_idx=0)
        x = torch.randn(2, 8, 4)
        full = model(x)
        torch.testing.assert_close(model(x, allowed=torch.ones(4, 4, dtype=torch.bool)), full)
        _, p = model(x, allowed=torch.eye(4, dtype=torch.bool), return_attention=True)
        torch.testing.assert_close(p, torch.eye(4).expand(2, 2, 4, 4))
        with self.assertRaisesRegex(ValueError, 'at least one'):
            model(x, allowed=torch.zeros(4, 4, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, 'boolean'):
            model(x, allowed=torch.ones(4, 4))

    def test_cache_requires_inference_and_clears_for_new_prompts(self):
        model = CausalSelfAttention(8, 2, layer_idx=0)
        x = torch.randn(1, 8, 2)
        cache = KVCache(4)
        with torch.no_grad(), self.assertRaisesRegex(ValueError, 'eval mode'):
            model(x, cache=cache)
        model.eval()
        with self.assertRaisesRegex(ValueError, 'no_grad'):
            model(x, cache=cache)
        with torch.no_grad():
            first = model(x, cache=cache)
            model(x, cache=cache)
            with self.assertRaisesRegex(ValueError, 'exceed n_max'):
                model(x, cache=cache)
            self.assertEqual(cache.length(0), 4)
            cache.clear()
            self.assertEqual(cache.length(0), 0)
            torch.testing.assert_close(model(x, cache=cache), first)

    def test_invalid_cache_batch_or_mask_does_not_append(self):
        model = CausalSelfAttention(8, 2, layer_idx=0).eval()
        cache = KVCache(8)
        with torch.no_grad():
            model(torch.randn(1, 8, 2), cache=cache)
            with self.assertRaisesRegex(ValueError, 'B, H, and d_k'):
                model(torch.randn(2, 8, 1), cache=cache)
            with self.assertRaisesRegex(ValueError, 'at least one'):
                model(torch.randn(1, 8, 1), cache=cache, allowed=torch.zeros(3, 1, dtype=torch.bool))
        self.assertEqual(cache.length(0), 2)

    def test_parameter_shapes_and_state_round_trip(self):
        model = CausalSelfAttention(8, 2, layer_idx=0)
        for head in model.heads:
            for linear in (head.query, head.key, head.value_down):
                self.assertEqual(linear.weight.shape, (4, 8))
                self.assertEqual(linear.bias.shape, (4, 1))
            self.assertEqual(head.value_up.weight.shape, (8, 4))
        copy = CausalSelfAttention(8, 2, layer_idx=0)
        copy.load_state_dict(model.state_dict())
        x = torch.randn(1, 8, 3)
        torch.testing.assert_close(model(x), copy(x), rtol=0, atol=0)

    def test_invalid_dimensions_and_positions(self):
        for d_emb, H in [(7, 2), (6, 2), (0, 1)]:
            with self.assertRaises(ValueError):
                CausalSelfAttention(d_emb, H, layer_idx=0)
        with self.assertRaisesRegex(ValueError, 'even d_k'):
            ModelConfig(V=10, d_emb=6, H=2)
        for n, start_pos in [(0, 0), (1, -1)]:
            with self.assertRaises(ValueError):
                causal_mask(n, start_pos=start_pos)
        with self.assertRaises(ValueError):
            RotaryPositionalEmbedding(4)(torch.randn(4, 2), start_pos=-1)


if __name__ == '__main__':
    unittest.main()
