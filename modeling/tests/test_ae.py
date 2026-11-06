import copy
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import train as train_mod  # noqa: E402
from ae import MaskedDAE, corrupt, masked_reconstruction_loss  # noqa: E402
from data import CLASS_ORDER, load_handoff  # noqa: E402
from preprocess import Preprocessor  # noqa: E402
from train import class_weights, feature_groups, load_config, set_seed, train_model  # noqa: E402


def tiny_config():
    cfg = load_config()
    cfg["pretrain"]["epochs"] = 2
    cfg["finetune"]["epochs"] = 3
    cfg["finetune"]["patience"] = 5
    return cfg


@pytest.fixture(scope="module")
def mock_tensors():
    ds = load_handoff("handoff/mock")
    pre = Preprocessor.from_schema(ds.schema).fit(ds.train.X)
    out = pre.transform(ds.val.X)
    return ds, pre, torch.from_numpy(out.X), torch.from_numpy(out.mask)


@pytest.fixture(scope="module")
def run():
    return train_model("handoff/mock", tiny_config(), verbose=False)


# ---------------------------------------------------------------- shapes
def test_output_shapes(mock_tensors):
    _, _, x, m = mock_tensors
    torch.manual_seed(0)
    model = MaskedDAE(x.shape[1]).eval()
    assert model.encode(x, m).shape == (160, 64)
    assert model.reconstruct(x, m).shape == x.shape
    assert model.classify(x, m).shape == (160, 4)
    recon, logits = model(x, m)
    assert recon.shape == x.shape and logits.shape == (160, 4)


def test_architecture_sizes(mock_tensors):
    _, _, x, _ = mock_tensors
    model = MaskedDAE(x.shape[1])
    enc = [l.out_features for l in model.encoder if isinstance(l, torch.nn.Linear)]
    head = [l.out_features for l in model.head if isinstance(l, torch.nn.Linear)]
    assert enc == [256, 128, 64] and head == [32, 4]
    assert model.encoder[0].in_features == 2 * x.shape[1]  # values + mask


# --------------------------------------------------- loss ignores missing
def test_loss_ignores_missing_target_cells():
    g = torch.Generator().manual_seed(0)
    recon, target = torch.randn(8, 5, generator=g), torch.randn(8, 5, generator=g)
    observed = (torch.rand(8, 5, generator=g) > 0.3).float()
    base = masked_reconstruction_loss(recon, target, observed)
    changed = target.clone()
    changed[observed == 0] = 1e6
    assert masked_reconstruction_loss(recon, changed, observed) == base
    changed[observed == 0] = float("nan")
    assert masked_reconstruction_loss(recon, changed, observed) == base


def test_missing_cell_value_does_not_change_model_loss(mock_tensors):
    """End to end: change the stored value of missing cells; reconstruction loss is identical."""
    _, _, x, m = mock_tensors
    assert (m == 0).any()
    torch.manual_seed(0)
    model = MaskedDAE(x.shape[1]).eval()
    x2 = x.clone()
    x2[m == 0] = 1234.5
    loss1 = masked_reconstruction_loss(model.reconstruct(x, m), x, m)
    loss2 = masked_reconstruction_loss(model.reconstruct(x2, m), x2, m)
    assert loss1.item() == loss2.item()


def test_corruption_only_hits_observed_cells(mock_tensors):
    _, pre, x, m = mock_tensors
    groups = torch.from_numpy(feature_groups(pre))
    xc, mc = corrupt(x, m, groups, 0.5, torch.Generator().manual_seed(0))
    assert ((mc == 1) <= (m == 1)).all()                   # never un-hides a missing cell
    assert torch.equal(mc[m == 0], m[m == 0])              # missing cells untouched
    hidden = (m == 1) & (mc == 0)
    assert 0.3 < hidden.sum() / (m == 1).sum() < 0.7      # ~ rate of observed cells
    assert (xc[hidden] == 0).all()
    # whole one-hot blocks are hidden together
    for gid in groups.unique():
        cols = (groups == gid).nonzero().flatten()
        block = mc[:, cols]
        assert ((block == block[:, :1]).all())


def test_class_weights_balanced():
    y = np.array([0] * 6 + [1] * 3 + [2] * 2 + [3] * 1)
    w = class_weights(y).numpy()
    np.testing.assert_allclose(w, 12 / (4 * np.array([6, 3, 2, 1])), rtol=1e-6)


# --------------------------------------------------------- training runs
def test_train_model_outputs(run):
    p = run.val_proba
    assert p.shape == (160, 4)
    assert list(p.columns) == list(CLASS_ORDER)
    assert list(p.index) == list(load_handoff("handoff/mock").val.participant_ids)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert 1 <= run.best_epoch <= 3


def test_same_seed_identical(run):
    again = train_model("handoff/mock", tiny_config(), verbose=False)
    pd.testing.assert_frame_equal(run.val_proba, again.val_proba, check_exact=True)
    for (k, a), (_, b) in zip(run.model.state_dict().items(), again.model.state_dict().items()):
        assert torch.equal(a, b), k
    assert run.history == again.history


def test_test_split_not_loaded(monkeypatch):
    """train_model asks the loader for train+val only, so no test row is ever in memory."""
    real = train_mod.load_handoff
    calls = []

    def spy(folder, splits=("train", "val", "test")):
        ds = real(folder, splits=splits)
        calls.append((tuple(splits), ds.test))
        return ds

    monkeypatch.setattr(train_mod, "load_handoff", spy)
    train_model("handoff/mock", tiny_config(), verbose=False)
    assert calls == [(("train", "val"), None)]


def test_test_split_values_do_not_matter(run):
    """Even if a Dataset carries a (corrupted) test split, results are unchanged."""
    ds = load_handoff("handoff/mock")
    X = ds.test.X.copy()
    X[ds.schema.numeric] = 1e6
    bad = replace(ds, test=replace(ds.test, X=X, y=np.zeros_like(ds.test.y)))
    again = train_mod.train_on_dataset(bad, tiny_config(), verbose=False)
    pd.testing.assert_frame_equal(run.val_proba, again.val_proba, check_exact=True)


def test_config_has_required_keys():
    cfg = load_config()
    assert cfg["model"]["hidden"] == [256, 128] and cfg["model"]["latent"] == 64
    for k in ("dropout", "activation", "head_hidden"):
        assert k in cfg["model"]
    for k in ("lr", "weight_decay", "epochs", "mask_rate"):
        assert k in cfg["pretrain"]
    for k in ("lr", "weight_decay", "epochs", "patience"):
        assert k in cfg["finetune"]
    assert "seed" in cfg
