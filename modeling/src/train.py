"""Train the masked denoising autoencoder + head (task B4).

Step 1 (pretrain): corrupt observed cells, reconstruct, Huber loss on cells observed before
                   corruption. Train rows only, no labels.
Step 2 (finetune): encoder + head with class-weighted cross-entropy (weights from train labels).
                   Early stopping on val macro-F1.
The test split is never used in this file.

    python modeling/src/train.py [--config modeling/configs/ae.yaml] [--handoff handoff/mock]
                                 [--pretrain-epochs N] [--finetune-epochs N]
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ae import MaskedDAE, corrupt, masked_reconstruction_loss  # noqa: E402
from data import CLASS_ORDER, DEV_SPLITS, load_handoff  # noqa: E402
from evaluate import balanced_accuracy, macro_f1  # noqa: E402
from preprocess import Preprocessor  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "ae.yaml"
N_CLASSES = len(CLASS_ORDER)


# ------------------------------------------------------------------- utilities
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True)


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text())


def feature_groups(pre: Preprocessor) -> np.ndarray:
    """Feature index for each output column, so a categorical's one-hot block is corrupted as one."""
    groups = list(range(len(pre.numeric)))
    for i, c in enumerate(pre.categorical, start=len(pre.numeric)):
        groups += [i] * len(pre.vocab_[c])
    return np.asarray(groups, dtype=np.int64)


def class_weights(y_train: np.ndarray) -> torch.Tensor:
    """Balanced weights n / (K * count_k), from train labels only."""
    counts = np.bincount(y_train, minlength=N_CLASSES).astype(np.float64)
    w = np.where(counts > 0, len(y_train) / (N_CLASSES * np.maximum(counts, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32)


def _batches(n: int, batch_size: int, gen: torch.Generator):
    perm = torch.randperm(n, generator=gen)
    for i in range(0, n, batch_size):
        yield perm[i:i + batch_size]


def _val_scores(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {"macro_f1": macro_f1(y, proba), "bal_acc": balanced_accuracy(y, proba)}


@torch.no_grad()
def predict_proba(model: MaskedDAE, x: torch.Tensor, m: torch.Tensor) -> np.ndarray:
    model.eval()
    return torch.softmax(model.classify(x, m), dim=1).double().numpy()


@dataclass
class TrainResult:
    model: MaskedDAE
    val_proba: pd.DataFrame           # (n_val, 4), columns = CLASS_ORDER, index = participant_id
    preprocessor: Preprocessor
    best_epoch: int
    best_val_macro_f1: float
    history: dict = field(default_factory=dict)


# ------------------------------------------------------------------ the steps
def pretrain(model, x_tr, m_tr, groups, cfg, gen, x_va=None, m_va=None, log=print):
    """Masked denoising pretraining on train rows. Labels are not used."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    hist = []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total, n = 0.0, 0
        for idx in _batches(len(x_tr), cfg["batch_size"], gen):
            x, m = x_tr[idx], m_tr[idx]
            xc, mc = corrupt(x, m, groups, cfg["mask_rate"], gen)
            loss = masked_reconstruction_loss(model.reconstruct(xc, mc), x, m, cfg["huber_delta"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            n += len(idx)
        rec = {"epoch": epoch, "train_recon": total / n}
        if x_va is not None:  # monitoring only; never used to fit or select
            model.eval()
            with torch.no_grad():
                rec["val_recon"] = masked_reconstruction_loss(
                    model.reconstruct(x_va, m_va), x_va, m_va, cfg["huber_delta"]).item()
        hist.append(rec)
        if epoch == 1 or epoch % 10 == 0 or epoch == cfg["epochs"]:
            log(f"[pretrain] epoch {epoch:3d}  train_recon={rec['train_recon']:.4f}"
                + (f"  val_recon={rec['val_recon']:.4f}" if "val_recon" in rec else ""))
    return hist


def finetune(model, x_tr, m_tr, y_tr, x_va, m_va, y_va, groups, cfg, mask_rate, huber_delta,
             gen, log=print):
    """Encoder + head, class-weighted CE; keeps the weights with the best val macro-F1."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    weights = class_weights(y_tr.numpy())
    best = {"f1": -1.0, "epoch": 0, "state": copy.deepcopy(model.state_dict())}
    hist, stale = [], 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total, n = 0.0, 0
        for idx in _batches(len(x_tr), cfg["batch_size"], gen):
            x, m, y = x_tr[idx], m_tr[idx], y_tr[idx]
            recon, logits = model(x, m)
            loss = F.cross_entropy(logits, y, weight=weights)
            if cfg.get("recon_weight", 0) > 0:
                xc, mc = corrupt(x, m, groups, mask_rate, gen)
                loss = loss + cfg["recon_weight"] * masked_reconstruction_loss(
                    model.reconstruct(xc, mc), x, m, huber_delta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            n += len(idx)
        scores = _val_scores(y_va.numpy(), predict_proba(model, x_va, m_va))
        hist.append({"epoch": epoch, "train_loss": total / n, **{f"val_{k}": v for k, v in scores.items()}})
        improved = scores["macro_f1"] > best["f1"]
        if improved:
            best = {"f1": scores["macro_f1"], "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or improved:
            log(f"[finetune] epoch {epoch:3d}  train_loss={total / n:.4f}  "
                f"val_macroF1={scores['macro_f1']:.4f}  val_balacc={scores['bal_acc']:.4f}"
                + ("  *best" if improved else ""))
        if stale >= cfg["patience"]:
            log(f"[finetune] early stop at epoch {epoch} (no val macro-F1 gain for {stale} epochs)")
            break
    model.load_state_dict(best["state"])
    return best, hist


# ------------------------------------------------------------------- entry
def train_model(handoff_dir: str | Path | None = None, config: dict | None = None,
                verbose: bool = True) -> TrainResult:
    """Fit on train only; val only for early stopping. Test rows are never loaded.
    Returns model + val probabilities (in a TrainResult)."""
    cfg = config if config is not None else load_config()
    handoff_dir = handoff_dir or cfg.get("handoff_dir", "handoff/mock")
    return train_on_dataset(load_handoff(handoff_dir, splits=DEV_SPLITS), cfg, verbose)


def train_on_dataset(ds, config: dict, verbose: bool = True,
                     features: list[str] | None = None) -> TrainResult:
    """Same as train_model but on an already-loaded Dataset. ds.test is never touched.
    features: restrict to these schema features; the Preprocessor is refitted on train."""
    cfg = copy.deepcopy(config)
    log = print if verbose else (lambda *a, **k: None)
    set_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    train, val = ds.train, ds.val
    pre = Preprocessor.from_schema(ds.schema, features).fit(train.X)
    t, v = pre.transform(train.X), pre.transform(val.X)
    x_tr, m_tr = torch.from_numpy(t.X), torch.from_numpy(t.mask)
    x_va, m_va = torch.from_numpy(v.X), torch.from_numpy(v.mask)
    y_tr, y_va = torch.from_numpy(train.y.copy()), torch.from_numpy(val.y.copy())
    groups = torch.from_numpy(feature_groups(pre))

    mc = cfg["model"]
    model = MaskedDAE(x_tr.shape[1], hidden=tuple(mc["hidden"]), latent=mc["latent"],
                      head_hidden=mc["head_hidden"], n_classes=N_CLASSES,
                      dropout=mc["dropout"], activation=mc["activation"])
    log(f"handoff: {ds.folder}")
    log(f"seed={cfg['seed']}  train={len(y_tr)}  val={len(y_va)}  inputs={x_tr.shape[1]}  "
        f"observed(train)={m_tr.mean().item():.3f}  params={sum(p.numel() for p in model.parameters())}")
    log(f"class weights (train): {[round(float(w), 3) for w in class_weights(train.y)]}")
    if "mock" in str(ds.folder.name):
        log("NOTE: mock data is random; numbers below are a smoke test, not results.")

    pc, fc = cfg["pretrain"], cfg["finetune"]
    pre_hist = pretrain(model, x_tr, m_tr, groups, pc, gen, x_va, m_va, log=log)
    best, ft_hist = finetune(model, x_tr, m_tr, y_tr, x_va, m_va, y_va, groups, fc,
                             pc["mask_rate"], pc["huber_delta"], gen, log=log)
    log(f"best val macro-F1 {best['f1']:.4f} at finetune epoch {best['epoch']}")

    proba = pd.DataFrame(predict_proba(model, x_va, m_va), columns=list(CLASS_ORDER),
                         index=pd.Index(val.participant_ids, name="participant_id"))
    return TrainResult(model=model, val_proba=proba, preprocessor=pre, best_epoch=best["epoch"],
                       best_val_macro_f1=best["f1"],
                       history={"pretrain": pre_hist, "finetune": ft_hist})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the masked DAE + head on a handoff folder.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--handoff", default=None)
    ap.add_argument("--pretrain-epochs", type=int)
    ap.add_argument("--finetune-epochs", type=int)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.pretrain_epochs is not None:
        cfg["pretrain"]["epochs"] = args.pretrain_epochs
    if args.finetune_epochs is not None:
        cfg["finetune"]["epochs"] = args.finetune_epochs
    train_model(args.handoff, cfg)
