"""Classifying hazard points against a footprint. 

"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tcmc.geo import nearest_idx

TC = 1
DEFAULT_CHUNK = 250_000

ASSOCIATED = "associated"
OTHER = "attributed_other"
NO_FOOTPRINT = "no_footprint"
OUT_OF_WINDOW = "out_of_window"
UNCLASSIFIABLE = "unclassifiable"

OUTCOMES = (ASSOCIATED, OTHER, NO_FOOTPRINT, OUT_OF_WINDOW, UNCLASSIFIABLE)

try:
    from skimage.segmentation import expand_labels as _expand
except ImportError:                                            # pragma: no cover
    _expand = None


# ══════════════════════════════════════════════════════════════════════════════
class Lookup:
    """A dilated label stack plus its axes.  Build once per footprint, reuse.

    `field` may be any integer label field on the storm grid: a lineage
    (1 = TC, >1 foreign), or a binary footprint written as 0/1, in which case
    `attributed_other` is identically zero -- which is correct for a parametric
    field, since an axisymmetric vortex has no foreign lineages by construction.
    """

    __slots__ = ("L", "ft", "lat", "lon", "dilate", "name")

    def __init__(self, field, *, dilate_cells: int = 1, name: str = ""):
        src = np.asarray(getattr(field, "values", field))
        self.L = np.empty(src.shape, np.int16)
        for k in range(src.shape[0]):
            f = src[k].astype(np.int32, copy=False)
            self.L[k] = _expand(f, distance=dilate_cells) \
                if (dilate_cells and _expand is not None) else f
        if hasattr(field, "coords"):
            self.ft = pd.to_datetime(field["valid_time"].values).to_numpy(
                "datetime64[ns]")
            self.lat = np.asarray(field["latitude"].values, float)
            self.lon = np.asarray(field["longitude"].values, float)
        else:
            self.ft = self.lat = self.lon = None
        self.dilate = int(dilate_cells)
        self.name = name

    @classmethod
    def from_arrays(cls, labels, times, lat, lon, *, dilate_cells: int = 1,
                    name: str = ""):
        obj = cls.__new__(cls)
        src = np.asarray(labels)
        obj.L = np.empty(src.shape, np.int16)
        for k in range(src.shape[0]):
            f = src[k].astype(np.int32, copy=False)
            obj.L[k] = _expand(f, distance=dilate_cells) \
                if (dilate_cells and _expand is not None) else f
        obj.ft = pd.to_datetime(np.asarray(times)).to_numpy("datetime64[ns]")
        obj.lat = np.asarray(lat, float)
        obj.lon = np.asarray(lon, float)
        obj.dilate = int(dilate_cells)
        obj.name = name
        return obj


# ══════════════════════════════════════════════════════════════════════════════
def classify_points(lons, lats, times, lu: Lookup, *, tol: str = "3h",
                    chunk: int = DEFAULT_CHUNK) -> dict:
    """Nearest-owned-frame lookup.  Peak memory O(chunk), independent of n and T.

    Returns arrays of length n: `in_box`, `in_window`, `lineage`, `dt_hours`.
    """
    lons = np.asarray(lons, float)
    lats = np.asarray(lats, float)
    n = lons.size
    out = dict(in_box=np.zeros(n, bool), in_window=np.zeros(n, bool),
               lineage=np.zeros(n, np.int16), dt_hours=np.full(n, np.nan))
    if n == 0 or lu.ft is None or lu.ft.size == 0:
        return out

    pt = pd.to_datetime(pd.Series(times)).to_numpy("datetime64[ns]")
    td = pd.Timedelta(tol).to_timedelta64()

    in_box = ((lons >= lu.lon.min()) & (lons <= lu.lon.max())
              & (lats >= lu.lat.min()) & (lats <= lu.lat.max()))
    # In-window is a statement about the FRAME AXIS, made independently of
    # whether anything was found: it is what separates "we did not look there"
    # from "we looked and there was nothing".
    in_window = (pt >= lu.ft.min() - td) & (pt <= lu.ft.max() + td)
    out["in_box"] = in_box
    out["in_window"] = in_window

    idx = np.flatnonzero(in_box & in_window)
    if idx.size == 0:
        return out

    last = lu.ft.size - 1
    for s0 in range(0, idx.size, chunk):
        sl = idx[s0:s0 + chunk]
        p = pt[sl]
        li = nearest_idx(lu.lat, lats[sl])
        lj = nearest_idx(lu.lon, lons[sl])
        lo = np.searchsorted(lu.ft, p - td, "left")
        hi = np.searchsorted(lu.ft, p + td, "right")
        W = int(np.max(hi - lo)) if sl.size else 0
        if W == 0:
            continue
        m = sl.size
        best = np.full(m, np.iinfo(np.int64).max)
        lin = np.zeros(m, np.int16)
        for w in range(W):                              # W ~ 9, not T
            k = np.minimum(lo + w, last)
            live = (lo + w) < hi
            v = lu.L[k, li, lj]
            d = np.abs((lu.ft[k] - p).astype("int64"))
            take = live & (v > 0) & (d < best)          # NEAREST owned frame
            if take.any():
                best[take] = d[take]
                lin[take] = v[take]
        out["lineage"][sl] = lin
        got = best < np.iinfo(np.int64).max
        out["dt_hours"][sl[got]] = best[got] / 3.6e12
    return out


def outcome_of(res: dict) -> np.ndarray:
    """Per-point outcome label.  The five-way split, computed in one place."""
    n = res["lineage"].size
    out = np.full(n, UNCLASSIFIABLE, dtype=object)
    box = res["in_box"]
    win = res["in_window"]
    out[box & ~win] = OUT_OF_WINDOW
    live = box & win
    out[live] = NO_FOOTPRINT
    out[live & (res["lineage"] == TC)] = ASSOCIATED
    out[live & (res["lineage"] > TC)] = OTHER
    return out


def classify(points: pd.DataFrame, lu: Lookup, *, tol: str = "3h",
             chunk: int = DEFAULT_CHUNK) -> pd.DataFrame:
    """`points` with lineage, dt_hours and outcome attached.  Order preserved.

    Order preservation matters: the pruned set is computed by aligning two
    classifications of the SAME point table positionally, so a reorder here
    would silently mispair them.
    """
    if points is None or not len(points):
        return points if points is not None else pd.DataFrame()
    r = classify_points(points["lon"], points["lat"], points["time"], lu,
                        tol=tol, chunk=chunk)
    out = points.copy()
    out["lineage"] = r["lineage"]
    out["dt_hours"] = r["dt_hours"]
    out["in_box"] = r["in_box"]
    out["in_window"] = r["in_window"]
    out["outcome"] = outcome_of(r)
    return out


def classify_by_hazard(points: pd.DataFrame, lu: Lookup, tolerances: dict,
                       *, chunk: int = DEFAULT_CHUNK) -> pd.DataFrame:
    """Classify a mixed point table, each hazard at its own tolerance.

    Necessary because the tolerances differ by an order of magnitude and for
    different reasons: +/-3 h for a minute-resolution tornado report is a field
    resolution allowance, +/-24 h for a date-only NFIP claim is measurement
    error, and +/-24 h for a flood report is physical lag.  Running them all at
    one tolerance would either over-associate the reports or under-associate the
    claims.
    """
    if points is None or not len(points):
        return points if points is not None else pd.DataFrame()
    parts = []
    for hz, sub in points.groupby("hazard", sort=False):
        parts.append(classify(sub, lu, tol=tolerances.get(str(hz), "3h"),
                              chunk=chunk))
    out = pd.concat(parts) if parts else points.copy()
    return out.loc[points.index]                       # restore the input order


# ══════════════════════════════════════════════════════════════════════════════
def counts(df: pd.DataFrame) -> dict:
    """The five-way split as a dict, for one (footprint, hazard) cell."""
    z = {k: 0 for k in OUTCOMES}
    z["total"] = 0
    if df is None or not len(df) or "outcome" not in df:
        return z
    vc = df["outcome"].value_counts()
    for k in OUTCOMES:
        z[k] = int(vc.get(k, 0))
    z["total"] = int(len(df))
    return z


def associated(df: pd.DataFrame) -> pd.DataFrame:
    """The rows a footprint owns.  What gets written to a link table."""
    if df is None or not len(df) or "outcome" not in df:
        return df if df is not None else pd.DataFrame()
    return df[df["outcome"] == ASSOCIATED]


def pruned(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Rows associated under `a` but not under `b`.  The flipped set.

    Aligned positionally, which is safe because `classify` preserves input
    order and both classifications are of the same point table.  Under the old
    layout this was persisted to a `pruned/` directory; it is now derivable
    from the long link tables as a set difference, so it is computed on demand
    rather than stored twice.
    """
    if a is None or not len(a):
        return a if a is not None else pd.DataFrame()
    flip = (a["outcome"].to_numpy() == ASSOCIATED) & \
           (b["outcome"].to_numpy() != ASSOCIATED)
    return a[flip]
