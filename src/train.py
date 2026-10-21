"""Configurable scratch training with document-safe data and validation.

Run from the project folder: .venv/bin/python -m src.train --help
Edit MODEL_DEFAULTS/TRAIN_DEFAULTS below, or override them on the command line.
Training uses all token positions and NO KV cache. No job starts on import.
"""

import argparse
from collections.abc import Callable
import json
import math
import random
import shutil
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import torch
from torch import Tensor
from torch.nn import functional as F

from .config import ModelConfig, TrainConfig
from .data import IGNORE_INDEX, TokenWindows, load_splits, split_fingerprint
from .model import TinyTransformer
from .tokenizer import Tokenizer

class ModelOptions(TypedDict, total=False):
    """Architecture options excluding V, which is determined by training text."""

    d_emb: int
    H: int
    L: int
    n_max: int
    d_ff: int | None
    dropout: float
    rope_base: float
    tie_embeddings: bool


# Each key has its own type; a heterogeneous dict cannot express this relationship.
MODEL_DEFAULTS: ModelOptions = {"d_emb": 64, "H": 4, "L": 2, "n_max": 64, "d_ff": None,
                               "dropout": 0.0, "rope_base": 10000.0, "tie_embeddings": True}
TRAIN_DEFAULTS = TrainConfig(B=16, n=64, learning_rate=3e-4, max_steps=200, device="cpu")
DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "data/gutenberg/manifest.json"


def language_model_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """(B, V, n) logits + (B, n) IDs -> scalar mean over valid next-token targets.

    loss = mean(-log p(correct next token)). Cross-entropy combines log-softmax
    and target selection stably; do not softmax logits first. Padding is ignored.
    """
    if logits.ndim != 3 or targets.shape != (logits.shape[0], logits.shape[2]):
        raise ValueError("Expected logits (B, V, n) and aligned targets (B, n).")
    if targets.dtype not in (torch.int32, torch.int64):
        raise ValueError("Targets must be integer token IDs.")
    if not bool((targets != IGNORE_INDEX).any()):
        raise ValueError("A loss needs at least one valid target.")
    return F.cross_entropy(logits, targets.long(), ignore_index=IGNORE_INDEX)  # Scalar.


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("Choose cpu, cuda, mps, or auto.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this process.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Apple MPS is unavailable in this process; use cpu here.")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def perplexity(loss: float) -> float:
    return math.exp(loss) if loss < 700 else float("inf")


def evaluate(model: TinyTransformer, data: TokenWindows, config: TrainConfig,
             device: torch.device) -> dict:
    """Fixed validation windows; token-weighted loss, no gradients/updates/cache."""
    was_training = model.training
    model.eval()
    total_loss, total_tokens, total_windows = 0.0, 0, 0
    try:
        with torch.no_grad():
            for inputs, targets in data.evaluation_batches(
                    config.B, max_batches=config.eval_batches, seed=config.seed+1):
                inputs, targets = inputs.to(device), targets.to(device)  # Each: (B, n).
                loss = language_model_loss(model(inputs), targets)  # Scalar; mean valid-token loss.
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite evaluation loss.")
                count = int((targets != IGNORE_INDEX).sum().item())
                total_loss += loss.item()*count
                total_tokens += count
                total_windows += len(inputs)
    finally:
        model.train(was_training)
    mean = total_loss/total_tokens
    return dict(loss=mean, perplexity=perplexity(mean), tokens=total_tokens, windows=total_windows)


def check_forward(model: TinyTransformer, inputs: Tensor) -> None:
    """Check finite logits and causal prefix invariance before optimization."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            original = model(inputs)  # (B, V, n).
            if not bool(torch.isfinite(original).all()):
                raise FloatingPointError("Non-finite initial logits.")
            if inputs.shape[-1] > 1:
                boundary = inputs.shape[-1]//2
                changed = inputs.clone()  # (B, n).
                changed[:, boundary:] = (changed[:, boundary:]+1) % model.config.V
                altered = model(changed)  # (B, V, n).
                torch.testing.assert_close(original[..., :boundary], altered[..., :boundary],
                                           atol=1e-5, rtol=1e-5)
    finally:
        model.train(was_training)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[TinyTransformer, Tokenizer, dict]:
    """Restore model/config/tokenizer for generation. Cache is intentionally absent."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = TinyTransformer(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, Tokenizer.from_state_dict(payload["tokenizer"]), payload


def export_model(checkpoint_path: str | Path, output_path: str | Path) -> Path:
    """Export weights/config/tokenizer without optimizer, paths, or run metadata.

    The tokenizer vocabulary still reflects the training text. A compact export
    is for inference only; use latest.pt with --resume to continue optimization.
    """
    source, destination = Path(checkpoint_path).resolve(), Path(output_path).resolve()
    if source == destination:
        raise ValueError("Export to a different file to preserve the training checkpoint.")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    vocabulary = {"tokens": payload["tokenizer"]["tokens"], "token_counts": {},
                  "document_token_counts": [], "total_characters": 0}
    destination.parent.mkdir(parents=True, exist_ok=True)
    _save(destination, {"format": "transformer-lab-model-v1",
                        "model_config": payload["model_config"],
                        "model_state": payload["model_state"],
                        "tokenizer": vocabulary})
    return destination


def _device_rng(device: torch.device) -> Tensor | None:
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device)
    if device.type == "mps":
        return torch.mps.get_rng_state()
    return None


def _save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)  # Replace only after serialization completes.


def run_training(manifest_path: str | Path, run_dir: str | Path, *,
                 model_options: ModelOptions | None = None, train_config: TrainConfig | None = None,
                 resume: str | Path | None = None, dry_run: bool = False,
                 test_final: bool = False, should_stop: Callable[[], bool] | None = None) -> dict:
    """Train or validate a setup; configurations, metrics, and checkpoints stay local."""
    saved = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    if saved is not None and "train_config" not in saved:
        raise ValueError("An inference export cannot resume training; choose a full latest.pt checkpoint.")
    config = train_config or (TrainConfig(**saved["train_config"]) if saved else TRAIN_DEFAULTS)
    if test_final and (config.overfit_one_batch or dry_run):
        raise ValueError("Final test evaluation is for completed corpus training, not debugging/dry-run.")
    if config.num_threads is not None:
        torch.set_num_threads(config.num_threads)
    device = resolve_device(config.device)
    seed_everything(config.seed)
    splits = load_splits(manifest_path)
    fingerprint = split_fingerprint(splits)
    tokenizer = Tokenizer(doc.text for doc in splits["train"])
    options = model_options if model_options is not None else MODEL_DEFAULTS
    model_config = (ModelConfig(**saved["model_config"]) if saved and model_options is None
                    else ModelConfig(V=tokenizer.V, **options))
    if config.n > model_config.n_max:
        raise ValueError("Training n must not exceed model n_max.")
    if saved:
        if saved["data_fingerprint"] != fingerprint or saved["tokenizer"] != tokenizer.state_dict():
            raise ValueError("Resume requires the same document splits and tokenizer mapping.")
        if saved["model_config"] != asdict(model_config):
            raise ValueError("Resume requires the same model architecture/configuration.")
        for name in ("B", "n", "seed", "learning_rate", "weight_decay", "grad_clip", "overfit_one_batch", "eval_batches"):
            if saved["train_config"][name] != getattr(config, name):
                raise ValueError(f"Resume cannot change training field {name}.")
        if config.max_steps <= saved["step"] and not dry_run:
            raise ValueError("Set max_steps greater than the saved step when resuming.")

    encoded = {name: [torch.tensor(tokenizer.encode(doc.text), dtype=torch.long) for doc in documents]
               for name, documents in splits.items()}  # One 1D CPU token tensor per document.
    datasets = {name: TokenWindows(documents, config.n) for name, documents in encoded.items()}
    summary = {name: dict(documents=len(splits[name]), windows=len(datasets[name]),
                          tokens=sum(len(ids) for ids in encoded[name]),
                          targets=sum(max(len(ids)-1, 0) for ids in encoded[name]),
                          unknown_tokens=sum(int((ids == tokenizer.unk_id).sum()) for ids in encoded[name]))
               for name in splits}
    run_dir = Path(run_dir).resolve()
    same_run = bool(resume and Path(resume).resolve().parent == run_dir)
    if run_dir.exists() and any(run_dir.iterdir()) and not same_run:
        raise FileExistsError("Run folder is not empty; choose a new --run-dir or resume its checkpoint.")
    run_dir.mkdir(parents=True, exist_ok=True)
    if resume and not same_run:
        previous_best = Path(resume).resolve().parent / "best.pt"
        if not previous_best.exists():
            raise ValueError("A resumed run needs its original best.pt for model selection.")
        shutil.copy2(previous_best, run_dir / "best.pt")

    model = TinyTransformer(model_config).to(device)
    decay = [p for name, p in model.named_parameters() if name.endswith("weight")]
    no_decay = [p for name, p in model.named_parameters() if not name.endswith("weight")]
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=config.learning_rate)
    batch_rng = torch.Generator().manual_seed(config.seed+2)  # Independent CPU window RNG.
    step, best_loss, bad_evaluations, tokens_seen = 0, float("inf"), 0, 0
    fixed_batch = datasets["train"].sample_batch(config.B, generator=batch_rng) if config.overfit_one_batch else None
    if saved:
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        step, best_loss = saved["step"], saved["best_val_loss"]
        bad_evaluations, tokens_seen = saved["bad_evaluations"], saved["tokens_seen"]
        fixed_batch = saved["fixed_batch"]
        batch_rng.set_state(saved["batch_rng_state"])
    probe = datasets["train"].batch([0])[0].to(device)  # (1, n), fixed reload/causality input.
    check_forward(model, probe)
    if saved:
        torch.set_rng_state(saved["torch_rng_state"])
        random.setstate(saved["python_rng_state"])
        if saved["device_type"] == device.type and saved["device_rng_state"] is not None:
            if device.type == "cuda":
                torch.cuda.set_rng_state(saved["device_rng_state"], device)
            elif device.type == "mps":
                torch.mps.set_rng_state(saved["device_rng_state"])

    def record(event: dict) -> None:
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event)+"\n")

    def checkpoint() -> dict:
        return dict(model_config=asdict(model_config), train_config=asdict(config),
                    model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                    tokenizer=tokenizer.state_dict(), data_fingerprint=fingerprint,
                    manifest_path=str(Path(manifest_path).resolve()), split_summary=summary,
                    documents=[dict(path=doc.path, split=doc.split, sha256=doc.sha256)
                               for documents in splits.values() for doc in documents],
                    step=step, best_val_loss=best_loss, bad_evaluations=bad_evaluations,
                    tokens_seen=tokens_seen, fixed_batch=fixed_batch,
                    batch_rng_state=batch_rng.get_state(), torch_rng_state=torch.get_rng_state(),
                    python_rng_state=random.getstate(), device_type=device.type,
                    device_rng_state=_device_rng(device), torch_version=str(torch.__version__))

    parameter_count = sum(p.numel() for p in model.parameters())
    metadata = dict(model=asdict(model_config), training=asdict(config), device=str(device),
                    parameters=parameter_count, data_fingerprint=fingerprint, splits=summary,
                    documents=[dict(path=doc.path, split=doc.split, sha256=doc.sha256)
                               for documents in splits.values() for doc in documents])
    (run_dir / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (run_dir / "tokenizer.json").write_text(json.dumps(tokenizer.state_dict(), ensure_ascii=False), encoding="utf-8")
    print(f"Run: {run_dir}\nDevice: {device}; parameters: {parameter_count:,}; V={tokenizer.V:,}")
    print("Splits:", summary)
    print(f"Uniform-predictor reference loss: log(V)={math.log(tokenizer.V):.4f}")
    print(f"Corpus training tokens / parameter: {summary['train']['tokens']/parameter_count:.3f}; "
          "repeated optimization steps do not create new text.")
    print("Mode:", "FIXED-BATCH DEBUGGING" if config.overfit_one_batch else "random training windows")
    initial = evaluate(model, datasets["val"], config, device)
    if not saved:
        best_loss = initial["loss"]
        _save(run_dir / "best.pt", checkpoint())
        _save(run_dir / "latest.pt", checkpoint())
    record(dict(event="validation", step=step, **initial))
    print(f"Initial validation loss={initial['loss']:.4f}; ppl={initial['perplexity']:.2f}; tokens={initial['tokens']}")
    started = time.perf_counter()
    stopped = False
    if not dry_run:
        model.train()
        first_step = step+1
        for next_step in range(first_step, config.max_steps+1):
            if should_stop is not None and should_stop():
                stopped = True
                record(dict(event="stopped", step=step))
                print("Stopping at a completed optimizer step; saving a resumable checkpoint.")
                break
            step = next_step
            step_start = time.perf_counter()
            inputs, targets = fixed_batch if fixed_batch is not None else datasets["train"].sample_batch(config.B, generator=batch_rng)
            inputs, targets = inputs.to(device), targets.to(device)  # Each: (B, n).
            before = next(model.parameters()).detach().clone() if step == first_step else None
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)  # (B, V, n); no cache or sampling during training.
            loss = language_model_loss(logits, targets)  # Scalar average over valid targets.
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite training loss at step {step}.")
            loss.backward()
            if step == first_step and any(p.grad is None for p in model.parameters()):
                raise RuntimeError("A model parameter is disconnected from the training loss.")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            if before is not None and torch.equal(before, next(model.parameters()).detach()):
                raise RuntimeError("The first optimizer step did not change the embedding weights.")
            count = int((targets != IGNORE_INDEX).sum().item())
            tokens_seen += count
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elif device.type == "mps":
                torch.mps.synchronize()
            seconds = time.perf_counter()-step_start
            event = dict(event="train", step=step, loss=loss.item(), grad_norm=grad_norm.item(),
                         tokens=count, tokens_seen=tokens_seen, seconds=seconds, tokens_per_second=count/seconds)
            record(event)
            if step == first_step or step % config.log_every == 0 or step == config.max_steps:
                print(f"step={step} train_loss={loss.item():.4f} grad_norm={grad_norm.item():.3f} "
                      f"tokens/s={count/seconds:.0f}")
            if step % config.eval_every == 0 or step == config.max_steps:
                validation = evaluate(model, datasets["val"], config, device)
                improved = validation["loss"] < best_loss
                if improved:
                    best_loss, bad_evaluations = validation["loss"], 0
                else:
                    bad_evaluations += 1
                record(dict(event="validation", step=step, **validation))
                print(f"validation step={step} loss={validation['loss']:.4f} "
                      f"ppl={validation['perplexity']:.2f}; best={best_loss:.4f}")
                if improved:
                    _save(run_dir / "best.pt", checkpoint())
                _save(run_dir / "latest.pt", checkpoint())
                if config.patience and bad_evaluations >= config.patience and not config.overfit_one_batch:
                    print("Early stopping: validation did not improve.")
                    break
        _save(run_dir / "latest.pt", checkpoint())

    # A fresh module must reproduce saved logits. Preserve CPU RNG during its random initialization.
    model.eval()
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        reference = model(probe)
        verification_path = Path(resume) if resume is not None and saved and dry_run else run_dir / "latest.pt"
        restored, restored_tokenizer, _ = load_checkpoint(verification_path, device)
        torch.testing.assert_close(restored(probe), reference, atol=1e-5, rtol=1e-5)
        if restored_tokenizer.state_dict() != tokenizer.state_dict():
            raise RuntimeError("Checkpoint tokenizer did not round-trip.")
    result: dict[str, Any] = dict(run_dir=str(run_dir), step=step, best_val_loss=best_loss, tokens_seen=tokens_seen,
                  seconds=time.perf_counter()-started, checkpoint_reload_verified=True, dry_run=dry_run,
                  initial_validation=initial, stopped=stopped)
    if test_final and not stopped:
        best_model, _, best_state = load_checkpoint(run_dir / "best.pt", device)
        result["test"] = evaluate(best_model, datasets["test"], replace(config, eval_batches=None), device)
        result["test_checkpoint_step"] = best_state["step"]
        record(dict(event="test", step=best_state["step"], **result["test"]))
        print("Final test (validation-selected checkpoint):", result["test"])
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Checkpoint reload verified. Test evaluated:", test_final)
    return result


def main(argv: list[str] | None = None) -> None:
    """CLI syntax lives here; dataclasses keep shared defaults/validation reusable."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-model", type=Path, metavar="CHECKPOINT", help="Export an inference model and exit.")
    parser.add_argument("--output", type=Path, help="Destination .pt file for --export-model.")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate data/model/checkpoint without optimizer updates.")
    parser.add_argument("--test-final", action="store_true", help="Evaluate the selected checkpoint on test after training.")
    for name in ("d_emb", "H", "L", "n_max", "d_ff"):
        parser.add_argument("--"+name.replace("_", "-"), dest=name, type=int)
    for name in ("dropout", "rope_base"):
        parser.add_argument("--"+name.replace("_", "-"), dest=name, type=float)
    parser.add_argument("--tie-embeddings", action=argparse.BooleanOptionalAction, default=None)
    for name in ("B", "n", "max_steps", "log_every", "eval_every", "eval_batches", "patience", "seed", "num_threads"):
        parser.add_argument("--"+name.replace("_", "-"), dest=name, type=int)
    for name in ("learning_rate", "weight_decay", "grad_clip"):
        parser.add_argument("--"+name.replace("_", "-"), dest=name, type=float)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"))
    parser.add_argument("--overfit-one-batch", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)
    if args.export_model:
        if args.output is None:
            parser.error("--export-model requires --output.")
        print(export_model(args.export_model, args.output))
        return
    if args.output is not None:
        parser.error("--output requires --export-model.")
    saved = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if saved is not None and "train_config" not in saved:
        parser.error("An inference export cannot resume training; choose a full latest.pt checkpoint.")
    # JSON/checkpoint/argparse values are dynamic at this boundary; ModelConfig
    # performs runtime validation before tensors are constructed.
    model_options: Any = {key: value for key, value in (saved["model_config"] if saved else MODEL_DEFAULTS).items() if key != "V"}
    training = dict(saved["train_config"]) if saved else asdict(TRAIN_DEFAULTS)
    for name in model_options:
        if getattr(args, name) is not None:
            model_options[name] = getattr(args, name)
    if args.d_emb is not None and args.d_ff is None and not saved:
        model_options["d_ff"] = None  # Keep the default expansion at four times the new d_emb.
    for name in training:
        if getattr(args, name) is not None:
            training[name] = getattr(args, name)
    manifest = args.manifest or (Path(saved["manifest_path"]) if saved else DEFAULT_MANIFEST)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    run_dir = args.run_dir or (args.resume.resolve().parent if args.resume else Path("runs") / f"train-{stamp}")
    run_training(manifest, run_dir, model_options=model_options, train_config=TrainConfig(**training),
                 resume=args.resume, dry_run=args.dry_run, test_final=args.test_final)


if __name__ == "__main__":
    main()
