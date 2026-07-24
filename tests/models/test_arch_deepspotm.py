"""Unit tests for the DeepSpot-M architecture wrapper.

These tests inject a lightweight fake ``deepspotm`` module so the tiatoolbox
wrapper (gene subsetting, tile preprocessing, output shape, the feature-extractor
``infer_batch`` contract) is exercised without the gated ``ratschlab/DeepSpotM``
weights or the optional ``deepspotm`` dependency.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch

from tiatoolbox.models.architecture.deepspotm import DeepSpotM

PANEL = ["EPCAM", "CD3D", "PTPRC", "COL1A1", "BRAF"]
RNG = np.random.default_rng(0)


def _make_fake_deepspotm() -> types.ModuleType:
    """Build a stand-in ``deepspotm`` module with the API the wrapper uses."""

    class _Transform:
        """Minimal eval transform: channel-last uint8 tile -> (3, 224, 224)."""

        def __call__(self: _Transform, tile: np.ndarray) -> torch.Tensor:
            arr = np.asarray(tile)[..., :3].astype("float32") / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1)

    class _FakeDeepSpotM(torch.nn.Module):
        """Fake DeepSpot-M mirroring the real forward / gene-indexing API."""

        def __init__(self: _FakeDeepSpotM) -> None:
            super().__init__()
            self.gene_names = list(PANEL)
            self._gene_to_idx = {g: i for i, g in enumerate(self.gene_names)}
            self.head = torch.nn.Linear(3 * 224 * 224, len(self.gene_names))

        def genes_to_indices(self: _FakeDeepSpotM, genes: list[str]) -> torch.Tensor:
            names = [genes] if isinstance(genes, str) else list(genes)
            missing = [g for g in names if g not in self._gene_to_idx]
            if missing:
                raise KeyError(missing)
            return torch.tensor([self._gene_to_idx[g] for g in names], dtype=torch.long)

        def forward(
            self: _FakeDeepSpotM,
            pixel_values: torch.Tensor,
            gene_indices: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, None, None]:
            expr = self.head(pixel_values.reshape(pixel_values.shape[0], -1))
            if gene_indices is not None:
                expr = expr.index_select(1, gene_indices.to(expr.device))
            return expr, None, None

        @classmethod
        def from_pretrained(
            cls: type[_FakeDeepSpotM],
            repo_id_or_path: str,
            *,
            source: str | None = None,
            device: str | None = None,
            revision: str | None = None,
        ) -> tuple[_FakeDeepSpotM, _Transform]:
            _ = (repo_id_or_path, source, device, revision)
            return cls(), _Transform()

    module = types.ModuleType("deepspotm")
    module.DeepSpotM = _FakeDeepSpotM
    return module


@pytest.fixture
def fake_deepspotm(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Register the fake ``deepspotm`` module for the duration of a test."""
    module = _make_fake_deepspotm()
    monkeypatch.setitem(sys.modules, "deepspotm", module)
    return module


def _tiles(n: int) -> np.ndarray:
    """Return ``n`` random NHWC uint8 RGB tiles."""
    return RNG.integers(0, 255, size=(n, 224, 224, 3), dtype=np.uint8)


@pytest.mark.usefixtures("fake_deepspotm")
def test_full_panel() -> None:
    """The full panel is loaded and infer_batch returns one array per batch."""
    model = DeepSpotM()
    assert model.gene_names == PANEL
    assert model.gene_indices is None

    n_tiles = 3
    output = DeepSpotM.infer_batch(model, _tiles(n_tiles), device="cpu")
    assert isinstance(output, list)
    assert len(output) == 1
    assert output[0].shape == (n_tiles, len(PANEL))


@pytest.mark.usefixtures("fake_deepspotm")
def test_gene_subset() -> None:
    """A gene subset restricts (and orders) the output columns."""
    genes = ["PTPRC", "EPCAM"]
    model = DeepSpotM(genes=genes)
    assert model.gene_names == genes

    n_tiles = 2
    output = DeepSpotM.infer_batch(model, _tiles(n_tiles), device="cpu")
    assert output[0].shape == (n_tiles, len(genes))


@pytest.mark.usefixtures("fake_deepspotm")
def test_single_gene_string() -> None:
    """A single gene symbol (str) is accepted and normalized to a 1-item panel."""
    model = DeepSpotM(genes="EPCAM")
    assert model.gene_names == ["EPCAM"]


@pytest.mark.usefixtures("fake_deepspotm")
def test_unknown_gene_raises() -> None:
    """An unknown gene symbol raises KeyError at construction time."""
    with pytest.raises(KeyError):
        DeepSpotM(genes=["NOT_A_GENE"])


@pytest.mark.usefixtures("fake_deepspotm")
def test_forward_preprocessed_tensor() -> None:
    """Passing a preprocessed tensor through forward returns the expression matrix."""
    genes = ["EPCAM", "CD3D", "PTPRC"]
    model = DeepSpotM(genes=genes)
    pixel_values = torch.rand(2, 3, 224, 224)
    expression = model(pixel_values)
    assert expression.shape == (2, len(genes))


@pytest.mark.usefixtures("fake_deepspotm")
def test_float_batch_is_coerced() -> None:
    """A float NHWC batch in [0, 255] is handled without rescaling artifacts."""
    model = DeepSpotM(genes=["EPCAM"])
    tiles = _tiles(2).astype("float32")
    output = DeepSpotM.infer_batch(model, tiles, device="cpu")
    assert output[0].shape == (2, 1)


def test_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """A helpful ImportError is raised when ``deepspotm`` is not installed."""
    monkeypatch.setitem(sys.modules, "deepspotm", None)
    with pytest.raises(ImportError, match="deepspotm"):
        DeepSpotM()
