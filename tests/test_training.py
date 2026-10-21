"""Data boundaries, loss, validation, checkpointing, and exact CPU resume tests."""

import contextlib
import hashlib
import io
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from src.config import ModelConfig, TrainConfig
from src.data import IGNORE_INDEX, TokenWindows, load_splits, make_batch
from src.model import TinyTransformer
from src.tokenizer import Tokenizer
from src.train import evaluate, language_model_loss, load_checkpoint, main, run_training


def fixture_corpus(root: Path) -> Path:
    entries = []
    texts = {"train": ["a b a b a b a b", "b a b a b a b a"],
             "validation": ["a b a b"], "test": ["b a b"]}
    for split, documents in texts.items():
        for index, text in enumerate(documents):
            path = root / split / f"doc{index}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            entries.append(dict(path=str(path.relative_to(root)), split=split,
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(dict(documents=entries)))
    return manifest


class DataAndLossTests(unittest.TestCase):
    def test_all_transitions_covered_once_and_never_cross_documents(self):
        data = TokenWindows([torch.tensor([1, 2, 3, 4, 5, 6]), torch.tensor([7, 8, 9])], n=4)
        pairs = []
        for inputs, targets in data.evaluation_batches(2):
            pairs.extend(zip(inputs[targets != IGNORE_INDEX].tolist(), targets[targets != IGNORE_INDEX].tolist()))
        self.assertEqual(pairs, [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8), (8, 9)])
        self.assertNotIn((6, 7), pairs)
        inputs, targets = data.batch([1])
        self.assertTrue(torch.equal(inputs, torch.tensor([[5, 0, 0, 0]])))
        self.assertTrue(torch.equal(targets, torch.tensor([[6, -100, -100, -100]])))

    def test_batch_sampling_is_reproducible(self):
        ids = torch.arange(30)
        first = make_batch(ids, 3, 4, generator=torch.Generator().manual_seed(4))
        second = make_batch(ids, 3, 4, generator=torch.Generator().manual_seed(4))
        for left, right in zip(first, second):
            self.assertTrue(torch.equal(left, right))
        valid = first[1] != IGNORE_INDEX
        self.assertTrue(torch.equal(first[1][valid], first[0][valid]+1))

    def test_uniform_loss_and_manual_target_probability(self):
        logits = torch.zeros(2, 5, 3, requires_grad=True)
        targets = torch.tensor([[0, 1, 2], [3, 4, -100]])
        loss = language_model_loss(logits, targets)
        self.assertAlmostEqual(loss.item(), math.log(5), places=6)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[1, :, 2], torch.zeros(5)))
        known = torch.tensor([[[math.log(0.2)], [math.log(0.8)]]])
        self.assertAlmostEqual(language_model_loss(known, torch.tensor([[1]])).item(), -math.log(0.8), places=6)
        with self.assertRaises(ValueError):
            language_model_loss(logits, torch.full((2, 3), IGNORE_INDEX))

    def test_manifest_detects_changed_or_duplicated_documents(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = fixture_corpus(Path(temporary))
            splits = load_splits(manifest)
            self.assertEqual([len(splits[s]) for s in ("train", "val", "test")], [2, 1, 1])
            value = json.loads(manifest.read_text())
            value["documents"].append(value["documents"][0])
            manifest.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'unique'):
                load_splits(manifest)
            value["documents"].pop()
            manifest.write_text(json.dumps(value))
            (Path(temporary) / value["documents"][0]["path"]).write_text('changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load_splits(manifest)

    def test_tokenizer_state_round_trip_keeps_mapping_and_counts(self):
        tokenizer = Tokenizer(["a b!", "b a?"])
        restored = Tokenizer.from_state_dict(json.loads(json.dumps(tokenizer.state_dict())))
        self.assertEqual(restored.state_dict(), tokenizer.state_dict())
        self.assertEqual(restored.decode(restored.encode('a b!')), 'a b!')
        self.assertEqual(restored.encode('new'), [0])

    def test_validation_is_token_weighted_and_does_not_change_training_rng_or_weights(self):
        torch.manual_seed(2)
        model = TinyTransformer(ModelConfig(V=5, d_emb=8, H=2, L=1, n_max=4, dropout=0.3))
        data = TokenWindows([torch.tensor([1, 2, 3, 4, 2, 1])], n=4)
        config = TrainConfig(B=1, n=4)
        state = torch.get_rng_state().clone()
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        result = evaluate(model, data, config, torch.device('cpu'))
        self.assertEqual(result['tokens'], 5)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        with torch.no_grad():
            model.eval()
            inputs, targets = data.batch([0, 1])
            expected = language_model_loss(model(inputs), targets).item()
        self.assertAlmostEqual(result['loss'], expected, places=6)
        for name, p in model.named_parameters():
            self.assertTrue(torch.equal(before[name], p))


class TrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = fixture_corpus(self.root)
        self.options = dict(d_emb=8, H=2, L=1, n_max=4, dropout=0.15)
        self.config = TrainConfig(B=2, n=4, max_steps=4, eval_every=2, log_every=2,
                                  learning_rate=0.01, patience=0, num_threads=1)
        self.old_threads = torch.get_num_threads()
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()

    def tearDown(self):
        self.output.__exit__(None, None, None)
        torch.set_num_threads(self.old_threads)
        self.temporary.cleanup()

    def run_job(self, folder, **kwargs):
        return run_training(self.manifest, self.root / folder, model_options=self.options,
                            train_config=kwargs.pop('train_config', self.config), **kwargs)

    def test_cpu_run_updates_weights_saves_config_and_keeps_test_unused(self):
        result = self.run_job('job-train')
        self.assertEqual(result['step'], 4)
        self.assertTrue(result['checkpoint_reload_verified'])
        self.assertNotIn('test', result)
        model, tokenizer, state = load_checkpoint(self.root / 'job-train/latest.pt')
        self.assertEqual(tokenizer.V, model.config.V)
        self.assertGreater(state['tokens_seen'], 0)
        self.assertLessEqual(state['tokens_seen'], 32)  # Tail padding does not count as data.
        self.assertEqual(state['documents'][0]['split'], 'train')
        self.assertTrue((self.root / 'job-train/best.pt').exists())
        events = [json.loads(line) for line in (self.root / 'job-train/metrics.jsonl').read_text().splitlines()]
        self.assertEqual(sum(event['event'] == 'test' for event in events), 0)
        self.assertEqual(sum(event['event'] == 'train' for event in events), 4)
        self.assertEqual(state['tokens_seen'], sum(event['tokens'] for event in events if event['event'] == 'train'))

    def test_resume_matches_uninterrupted_cpu_training_including_dropout(self):
        self.run_job('whole')
        self.run_job('resumed', train_config=replace(self.config, max_steps=2))
        self.run_job('resumed', resume=self.root / 'resumed/latest.pt')
        left = torch.load(self.root / 'whole/latest.pt', weights_only=True)
        right = torch.load(self.root / 'resumed/latest.pt', weights_only=True)
        self.assertEqual(left['tokens_seen'], right['tokens_seen'])
        for name in left['model_state']:
            torch.testing.assert_close(left['model_state'][name], right['model_state'][name], rtol=0, atol=0)
        self.assertTrue(torch.equal(left['batch_rng_state'], right['batch_rng_state']))

    def test_resume_rejects_changed_corpus_and_config(self):
        self.run_job('resume')
        with self.assertRaisesRegex(ValueError, 'field B'):
            self.run_job('resume', train_config=replace(self.config, B=3, max_steps=6),
                         resume=self.root / 'resume/latest.pt')
        value = json.loads(self.manifest.read_text())
        value['documents'][0]['split'], value['documents'][2]['split'] = 'validation', 'train'
        self.manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'same document splits'):
            self.run_job('resume', train_config=replace(self.config, max_steps=6),
                         resume=self.root / 'resume/latest.pt')

    def test_dry_run_does_not_update_and_refuses_overwriting_an_existing_run(self):
        result = self.run_job('dry', dry_run=True)
        self.assertEqual(result['step'], 0)
        self.assertEqual(result['tokens_seen'], 0)
        with self.assertRaises(FileExistsError):
            self.run_job('dry', dry_run=True)

    def test_explicit_final_test_uses_validation_selected_checkpoint(self):
        result = self.run_job('job-test', test_final=True)
        self.assertEqual(result['test']['tokens'], 4)
        state = torch.load(self.root / 'job-test/best.pt', weights_only=True)
        self.assertEqual(result['test_checkpoint_step'], state['step'])

    def test_fixed_batch_debug_run_learns(self):
        result = self.run_job('overfit', train_config=replace(self.config, max_steps=30,
                                                             overfit_one_batch=True, eval_every=30))
        events = [json.loads(line) for line in (self.root / 'overfit/metrics.jsonl').read_text().splitlines()]
        losses = [event['loss'] for event in events if event['event'] == 'train']
        self.assertLess(losses[-1], losses[0]*0.6)
        self.assertEqual(result['step'], 30)

    def test_early_stopping_keeps_best_and_latest_distinct(self):
        initial = dict(loss=1.0, perplexity=math.e, tokens=8, windows=2)
        worse = dict(initial, loss=1.1)
        with patch('src.train.evaluate', side_effect=[initial, worse]):
            result = self.run_job('early-stop', train_config=replace(self.config, max_steps=10, patience=1))
        self.assertEqual(result['step'], 2)
        best = torch.load(self.root / 'early-stop/best.pt', weights_only=True)
        latest = torch.load(self.root / 'early-stop/latest.pt', weights_only=True)
        self.assertEqual(best['step'], 0)
        self.assertEqual(latest['step'], 2)

    def test_cli_syntax_creates_a_small_dry_run(self):
        main(['--manifest', str(self.manifest), '--run-dir', str(self.root / 'cli'),
              '--dry-run', '--B', '2', '--n', '4', '--d-emb', '8', '--H', '2', '--L', '1'])
        saved = json.loads((self.root / 'cli/config.json').read_text())
        self.assertEqual(saved['model']['d_emb'], 8)
        self.assertEqual(saved['model']['d_ff'], 32)


if __name__ == '__main__':
    unittest.main()
