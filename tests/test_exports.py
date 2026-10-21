"""Portable inference exports preserve predictions without resume metadata."""

import contextlib
import io
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import torch

from src.config import ModelConfig
from src.generate import main as generate_main
from src.model import TinyTransformer
from src.tokenizer import Tokenizer
from src.train import export_model, load_checkpoint, run_training


class ExportTests(unittest.TestCase):
    def test_export_omits_training_metadata_and_preserves_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer = Tokenizer("a b a b")
            config = ModelConfig(V=tokenizer.V, d_emb=8, H=2, L=1, n_max=8)
            model = TinyTransformer(config).eval()
            checkpoint = root / "latest.pt"
            torch.save(dict(model_config=asdict(config), model_state=model.state_dict(),
                            tokenizer=tokenizer.state_dict(), manifest_path="private/path",
                            documents=["private-name.txt"], optimizer_state={}, train_config={},
                            torch_rng_state=torch.get_rng_state()), checkpoint)
            exported = export_model(checkpoint, root / "model.pt")
            restored, restored_tokenizer, payload = load_checkpoint(exported)
            self.assertEqual(set(payload), {"format", "model_config", "model_state", "tokenizer"})
            self.assertEqual(restored_tokenizer.token_counts, {})
            self.assertEqual(restored_tokenizer.document_token_counts, [])
            self.assertEqual(restored_tokenizer.total_characters, 0)
            self.assertEqual(restored_tokenizer.token_to_id, tokenizer.token_to_id)
            ids = torch.tensor([tokenizer.encode("a b")])
            with torch.no_grad():
                torch.testing.assert_close(model(ids), restored(ids))
            with self.assertRaisesRegex(ValueError, "different file"):
                export_model(checkpoint, checkpoint)
            with self.assertRaisesRegex(ValueError, "cannot resume"):
                run_training(root / "absent.json", root / "new", resume=exported)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                generate_main(["--checkpoint", str(exported), "--prompt", "a b",
                               "--max-new-tokens", "2", "--top-k", "1"])
            self.assertTrue(output.getvalue().startswith("a b"))


if __name__ == "__main__":
    unittest.main()
