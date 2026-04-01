"""Join TPC and PMT event batches on (run, subrun, event) keys.

In the SBN experiment the TPC and PMT sub-detectors write separate trees (or
files).  ``EventJoiner`` performs an inner join so that the fusion model always
receives matched pairs.
"""

from __future__ import annotations

import logging
from typing import Optional

import awkward as ak
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical event-ID fields shared by all SBN ntuples.
EVENT_KEY_FIELDS = ("run", "subrun", "event")


class EventJoiner:
    """Inner-join TPC and PMT batches on (run, subrun, event).

    Parameters
    ----------
    tpc_key_fields:
        Branch names that form the event key in TPC batches.
    pmt_key_fields:
        Branch names that form the event key in PMT batches.
    """

    def __init__(
        self,
        tpc_key_fields: tuple[str, ...] = EVENT_KEY_FIELDS,
        pmt_key_fields: tuple[str, ...] = EVENT_KEY_FIELDS,
    ) -> None:
        self.tpc_key_fields = tpc_key_fields
        self.pmt_key_fields = pmt_key_fields

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def join(
        self,
        tpc_batch: ak.Array,
        pmt_batch: ak.Array,
    ) -> tuple[ak.Array, ak.Array]:
        """Return aligned (tpc, pmt) sub-arrays containing only matched events.

        Parameters
        ----------
        tpc_batch:
            Awkward array of TPC events with fields ``run``, ``subrun``, ``event``.
        pmt_batch:
            Awkward array of PMT events with the same key fields.

        Returns
        -------
        tpc_matched, pmt_matched:
            Sub-arrays restricted to events present in *both* batches, sorted by
            (run, subrun, event) so indices correspond one-to-one.
        """
        tpc_keys = self._extract_keys(tpc_batch, self.tpc_key_fields)
        pmt_keys = self._extract_keys(pmt_batch, self.pmt_key_fields)

        tpc_df = pd.DataFrame(tpc_keys, columns=list(self.tpc_key_fields))
        pmt_df = pd.DataFrame(pmt_keys, columns=list(self.pmt_key_fields))

        tpc_df["_tpc_idx"] = np.arange(len(tpc_df))
        pmt_df["_pmt_idx"] = np.arange(len(pmt_df))

        merged = tpc_df.merge(pmt_df, on=list(self.tpc_key_fields), how="inner")
        merged = merged.sort_values(list(self.tpc_key_fields)).reset_index(drop=True)

        tpc_idx = merged["_tpc_idx"].to_numpy()
        pmt_idx = merged["_pmt_idx"].to_numpy()

        n_tpc = len(tpc_batch)
        n_pmt = len(pmt_batch)
        n_matched = len(merged)
        logger.debug(
            "EventJoiner: tpc=%d, pmt=%d -> matched=%d", n_tpc, n_pmt, n_matched
        )

        return tpc_batch[tpc_idx], pmt_batch[pmt_idx]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_keys(
        batch: ak.Array,
        fields: tuple[str, ...],
    ) -> np.ndarray:
        """Stack key fields into a (N, len(fields)) int32 array."""
        columns = [ak.to_numpy(batch[f]).astype(np.int32) for f in fields]
        return np.column_stack(columns)
