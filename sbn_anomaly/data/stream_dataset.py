"""PyTorch IterableDataset that streams TPC features from ROOT files.

Two feature-extraction modes are supported:

**Waveform mode** (default)
    Each event in ``tpc_waveform`` is a variable-length nested array (channels
    × ticks).  The entire event payload is flattened to a 1-D array and then
    **padded** (with zeros) or **truncated** to a fixed ``input_dim`` so the
    model receives constant-size tensors.

**Hit mode** (``waveform_branch=None``)
    Higher-level hit quantities (e.g. ``hit_integral``, ``hit_charge``,
    ``hit_amplitude``) are read from the branches listed in ``hit_branches``.
    Values from all listed branches are concatenated per event in order and
    then padded/truncated to ``input_dim``.

Usage::

    # Waveform mode
    from sbn_anomaly.data.stream_dataset import TPCStreamDataset
    from torch.utils.data import DataLoader

    ds = TPCStreamDataset(
        file_paths=["/data/run.root"],
        tree_name="sbn_tree",
        waveform_branch="tpc_waveform",
        input_dim=256,
    )
    loader = DataLoader(ds, batch_size=256)

    # Hit mode
    ds = TPCStreamDataset(
        file_paths=["/data/run.root"],
        tree_name="sbn_tree",
        waveform_branch=None,
        hit_branches=["hit_integral", "hit_charge", "hit_amplitude"],
        input_dim=256,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import torch
from torch.utils.data import IterableDataset

from sbn_anomaly.data.streaming import RootStreamer

logger = logging.getLogger(__name__)


def extract_tpc_features(
    raw: np.ndarray,
    input_dim: int = 256,
) -> np.ndarray:
    """Flatten *raw* waveform values and pad/truncate to *input_dim*.

    Parameters
    ----------
    raw:
        1-D float32 array containing concatenated waveform samples for one
        event (already flattened from a nested awkward array).
    input_dim:
        Target feature vector length.

    Returns
    -------
    numpy.ndarray of shape ``(input_dim,)`` and dtype ``float32``.
    """
    raw = raw.astype(np.float32)
    if len(raw) >= input_dim:
        return raw[:input_dim]
    padded = np.zeros(input_dim, dtype=np.float32)
    padded[: len(raw)] = raw
    return padded


def extract_hit_features(
    event_data: dict,
    hit_branches: list,
    input_dim: int = 256,
) -> np.ndarray:
    """Concatenate per-event hit quantities and pad/truncate to *input_dim*.

    Values from each branch in *hit_branches* are flattened and concatenated
    in order (e.g. all ``hit_integral`` values, then all ``hit_charge`` values,
    etc.).  The resulting 1-D array is padded with zeros or truncated to
    produce a fixed-size feature vector.

    Parameters
    ----------
    event_data:
        Mapping from branch name to a 1-D float32 numpy array of per-hit
        values for a single event (already flattened from awkward arrays).
    hit_branches:
        Ordered list of branch names to include.
    input_dim:
        Target feature vector length.

    Returns
    -------
    numpy.ndarray of shape ``(input_dim,)`` and dtype ``float32``.
    """
    parts = [
        np.asarray(event_data[b], dtype=np.float32).ravel()
        for b in hit_branches
        if b in event_data
    ]
    raw = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    if len(raw) >= input_dim:
        return raw[:input_dim]
    padded = np.zeros(input_dim, dtype=np.float32)
    padded[: len(raw)] = raw
    return padded


class TPCStreamDataset(IterableDataset):
    """Stream per-event TPC feature vectors directly from ROOT files.

    Two input modes are supported:

    * **Waveform mode** (default): set ``waveform_branch`` to the name of the
      raw waveform branch.  The nested per-event array is flattened and
      padded/truncated to ``input_dim``.
    * **Hit mode**: set ``waveform_branch=None`` and provide a list of scalar
      hit-level branches via ``hit_branches`` (e.g. ``["hit_integral",
      "hit_charge", "hit_amplitude"]``).  Values from each branch are
      concatenated per event and padded/truncated to ``input_dim``.

    Parameters
    ----------
    file_paths:
        One or more paths to ``.root`` files.  Glob strings should be
        expanded by the caller before passing here.
    tree_name:
        Name of the TTree inside each ROOT file.
    waveform_branch:
        Branch name that holds the per-event waveform data.  Set to ``None``
        to disable waveform input and use ``hit_branches`` instead.
    hit_branches:
        Ordered list of hit-level scalar branches to use when
        ``waveform_branch`` is ``None``.  Ignored in waveform mode.
    branches:
        Full list of branches to load per event.  If ``None``, defaults to
        ``[waveform_branch]`` in waveform mode or ``hit_branches`` in hit
        mode.
    input_dim:
        Size of the output feature vector for each event (default 256).
    batch_size:
        Number of events per ``RootStreamer`` batch (controls memory use).
    normalize:
        When ``True``, apply per-event z-score standardisation (zero mean,
        unit variance).  Events with zero variance are left unchanged.
    max_events:
        Stop after yielding this many events (``None`` = unlimited).
    """

    def __init__(
        self,
        file_paths: Union[str, Path, Iterable[Union[str, Path]]],
        tree_name: str = "sbn_tree",
        waveform_branch: Optional[str] = "tpc_waveform",
        hit_branches: Optional[list] = None,
        branches: Optional[list] = None,
        input_dim: int = 256,
        batch_size: int = 512,
        normalize: bool = False,
        max_events: Optional[int] = None,
    ) -> None:
        if waveform_branch is None and not hit_branches:
            raise ValueError(
                "Either waveform_branch must be set, or hit_branches must be "
                "provided (waveform_branch=None selects hit mode)."
            )
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        self.file_paths = [Path(p) for p in file_paths]
        self.tree_name = tree_name
        self.waveform_branch = waveform_branch
        self.hit_branches: list = hit_branches or []
        # Determine which branches to actually request from the streamer.
        if branches is not None:
            self.branches = branches
        elif waveform_branch is not None:
            self.branches = [waveform_branch]
        else:
            self.branches = list(self.hit_branches)
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.normalize = normalize
        self.max_events = max_events

    # ------------------------------------------------------------------
    # IterableDataset interface
    # ------------------------------------------------------------------

    def __iter__(self):
        import awkward as ak

        streamer = RootStreamer(
            file_paths=self.file_paths,
            tree_name=self.tree_name,
            branches=self.branches,
            batch_size=self.batch_size,
        )

        n_yielded = 0
        for batch in streamer.stream():
            for i in range(len(batch)):
                if self.waveform_branch is not None:
                    raw = ak.to_numpy(
                        ak.flatten(batch[self.waveform_branch][i], axis=None)
                    ).astype(np.float32)
                    feat = extract_tpc_features(raw, self.input_dim)
                else:
                    event_data = {
                        b: ak.to_numpy(
                            ak.flatten(batch[b][i], axis=None)
                        ).astype(np.float32)
                        for b in self.hit_branches
                    }
                    feat = extract_hit_features(event_data, self.hit_branches, self.input_dim)

                if self.normalize:
                    mean = feat.mean()
                    std = feat.std()
                    if std > 0.0:
                        feat = (feat - mean) / std

                yield (torch.from_numpy(feat),)

                n_yielded += 1
                if self.max_events is not None and n_yielded >= self.max_events:
                    return
