"""Model wiring, shared weights, cached decoding, and top-k generation checks."""

import unittest

import torch
from torch.nn import functional as F

from src.cache import KVCache
from src.config import ModelConfig
from src.generate import generate, sample_top_k
from src.model import TinyTransformer, TransformerBlock


def small_model(*, tie_embeddings=True, n_max=8, dropout=0.0):
    return TinyTransformer(ModelConfig(V=13, d_emb=8, H=2, L=2, d_ff=16,
                                       n_max=n_max, tie_embeddings=tie_embeddings,
                                       dropout=dropout)).double()


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)

    def test_block_order_and_zero_updates_preserve_residual(self):
        block = TransformerBlock(ModelConfig(V=13, d_emb=8, H=2), layer_idx=0).double()
        order = []
        hooks = [module.register_forward_hook(lambda module, inputs, output, name=name: order.append(name))
                 for name, module in [('norm_attention', block.norm_attention),
                                      ('attention', block.attention),
                                      ('norm_ffn', block.norm_ffn), ('ffn', block.ffn)]]
        x = torch.randn(2, 8, 5, dtype=torch.double)
        block(x)
        self.assertEqual(order, ['norm_attention', 'attention', 'norm_ffn', 'ffn'])
        for hook in hooks:
            hook.remove()
        with torch.no_grad():
            for module in (block.attention, block.ffn):
                for parameter in module.parameters():
                    parameter.zero_()
        torch.testing.assert_close(block(x), x, atol=0, rtol=0)

    def test_logits_at_all_positions_and_last_column(self):
        model = small_model()
        for n in (1, 5, model.config.n_max):
            ids = torch.randint(0, model.config.V, (2, n))
            full = model(ids)
            self.assertEqual(full.shape, (2, model.config.V, n))
            last = model(ids, last_token_only=True)
            torch.testing.assert_close(last, full[..., -1:])
        self.assertEqual([block.attention.layer_idx for block in model.blocks], [0, 1])
        self.assertIsInstance(model.blocks, torch.nn.ModuleList)

    def test_tied_weight_is_shared_and_saves_one_table(self):
        tied = small_model()
        untied = small_model(tie_embeddings=False)
        self.assertEqual(tied.unembedding_weight.data_ptr(), tied.token_embedding.weight.data_ptr())
        self.assertNotEqual(untied.unembedding_weight.data_ptr(), untied.token_embedding.weight.data_ptr())
        self.assertEqual(sum(p.numel() for p in untied.parameters()) - sum(p.numel() for p in tied.parameters()),
                         tied.config.V * tied.config.d_emb)

    def test_tied_gradient_adds_lookup_and_unembedding_contributions(self):
        tied = small_model()
        untied = small_model(tie_embeddings=False)
        untied.load_state_dict(tied.state_dict(), strict=False)
        with torch.no_grad():
            untied.unembedding.weight.copy_(tied.token_embedding.weight.T)
        ids = torch.randint(0, tied.config.V, (2, 5))
        targets = torch.randint(0, tied.config.V, (2, 5))
        torch.testing.assert_close(tied(ids), untied(ids))
        F.cross_entropy(tied(ids), targets).backward()
        F.cross_entropy(untied(ids), targets).backward()
        expected = untied.token_embedding.weight.grad + untied.unembedding.weight.grad.T
        torch.testing.assert_close(tied.token_embedding.weight.grad, expected, atol=1e-12, rtol=1e-12)
        for parameter in tied.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_model_causality_and_batch_independence(self):
        model = small_model(dropout=0.25).eval()
        ids = torch.randint(0, model.config.V, (2, 6))
        changed = ids.clone()
        changed[0, 3:] = (changed[0, 3:] + 1) % model.config.V
        original, updated = model(ids), model(changed)
        torch.testing.assert_close(original[0, :, :3], updated[0, :, :3], atol=0, rtol=0)
        torch.testing.assert_close(original[1], updated[1], atol=0, rtol=0)

    def test_cached_model_matches_full_pass_and_mlp_only_sees_new_tokens(self):
        model = small_model(dropout=0.25).eval()
        ids = torch.randint(0, model.config.V, (2, 7))
        cache = KVCache(8)
        lengths = [[] for _ in model.blocks]
        hooks = [block.ffn.register_forward_pre_hook(
            lambda module, inputs, ell=ell: lengths[ell].append(inputs[0].shape[-1]))
                 for ell, block in enumerate(model.blocks)]
        with torch.no_grad():
            full = model(ids)
            for recorded in lengths:
                recorded.clear()
            pieces = [model(ids[:, :3], cache=cache)]
            for i in range(3, 7):
                pieces.append(model(ids[:, i:i+1], cache=cache))
        torch.testing.assert_close(torch.cat(pieces, dim=-1), full, atol=1e-12, rtol=1e-12)
        self.assertEqual(lengths, [[3, 1, 1, 1, 1], [3, 1, 1, 1, 1]])
        self.assertEqual([cache.length(ell) for ell in range(2)], [7, 7])
        for hook in hooks:
            hook.remove()

    def test_cached_model_handles_multi_token_chunks(self):
        model = small_model().eval()
        ids = torch.randint(0, model.config.V, (2, 7))
        cache = KVCache(8)
        with torch.no_grad():
            full = model(ids)
            parts = [model(ids[:, :2], cache=cache), model(ids[:, 2:5], cache=cache),
                     model(ids[:, 5:], cache=cache)]
        torch.testing.assert_close(torch.cat(parts, dim=-1), full, atol=1e-12, rtol=1e-12)

    def test_invalid_ids_lengths_and_incomplete_caches(self):
        model = small_model().eval()
        for ids in (torch.zeros(1, 9, dtype=torch.long), torch.tensor([[-1]]),
                    torch.tensor([[13]]), torch.zeros(1, 2), torch.empty(1, 0, dtype=torch.long)):
            with self.assertRaises(ValueError):
                model(ids)
        cache = KVCache(8)
        with torch.no_grad():
            model(torch.tensor([[1, 2]]), cache=cache)
            cache.layers.pop(1)
            with self.assertRaisesRegex(ValueError, 'same cached prefix length'):
                model(torch.tensor([[3]]), cache=cache)
        self.assertEqual(cache.length(0), 2)

    def test_generation_cached_and_uncached_match_across_context_rebuilds(self):
        model = small_model(n_max=4)
        prefix = torch.tensor([[1, 2, 3], [3, 2, 1]])
        for top_k in (1, 4):
            cached = generate(model, prefix, 8, top_k=top_k,
                              generator=torch.Generator().manual_seed(14))
            uncached = generate(model, prefix, 8, top_k=top_k, use_cache=False,
                                generator=torch.Generator().manual_seed(14))
            self.assertTrue(torch.equal(cached, uncached))
            self.assertTrue(torch.equal(cached[:, :3], prefix))
            self.assertEqual(cached.shape, (2, 11))
            self.assertTrue(model.training)

    def test_generation_prefill_then_only_single_tokens_and_restores_mode(self):
        model = small_model().eval()
        seen = []
        hook = model.token_embedding.register_forward_pre_hook(
            lambda module, inputs: seen.append(inputs[0].shape[-1]))
        prefix = torch.tensor([[1, 2, 3]])
        output = generate(model, prefix, 4, top_k=3)
        self.assertEqual(seen, [3, 1, 1, 1])
        self.assertFalse(model.training)
        self.assertEqual(output.shape, (1, 7))
        hook.remove()
        torch.testing.assert_close(generate(model, prefix, 0), prefix)

    def test_long_prompt_is_cropped_for_computation_but_preserved_in_output(self):
        model = small_model(n_max=4)
        prefix = torch.tensor([[1, 2, 3, 4, 5, 6]])
        cached = generate(model, prefix, 3, top_k=1)
        uncached = generate(model, prefix, 3, top_k=1, use_cache=False)
        self.assertTrue(torch.equal(cached, uncached))
        self.assertTrue(torch.equal(cached[:, :6], prefix))

    def test_top_k_restricts_candidates_and_samples_their_distribution(self):
        logits = torch.tensor([[0., 2., 1., 3.]])
        torch.testing.assert_close(sample_top_k(logits, top_k=1), torch.tensor([[3]]))
        draws = sample_top_k(logits.expand(5000, -1), top_k=2,
                             generator=torch.Generator().manual_seed(42))
        self.assertTrue(set(draws.flatten().tolist()) <= {1, 3})
        expected_probability = torch.sigmoid(torch.tensor(1.)).item()
        self.assertAlmostEqual((draws == 3).float().mean().item(), expected_probability, delta=0.03)
        self.assertEqual(sample_top_k(logits, top_k=100).shape, (1, 1))
        torch.testing.assert_close(sample_top_k(logits, top_k=2, temperature=1e-12), torch.tensor([[3]]))

    def test_sampling_rejects_invalid_inputs(self):
        for temperature in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                sample_top_k(torch.ones(1, 3), temperature=temperature)
        with self.assertRaises(ValueError):
            sample_top_k(torch.ones(1, 3), top_k=0)
        with self.assertRaises(ValueError):
            sample_top_k(torch.tensor([[float('nan'), 0.]]))

    def test_model_state_reload_reproduces_logits(self):
        model, restored = small_model(), small_model()
        restored.load_state_dict(model.state_dict())
        ids = torch.tensor([[1, 2, 3]])
        torch.testing.assert_close(model(ids), restored(ids), atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
