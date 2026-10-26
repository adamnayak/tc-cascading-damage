"""Dominant corridor arbitration

"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import ndimage as ndi

from tcmc.corridor.thr import ST, disk

MEASURES = ("content", "peak", "mean", "flux")

#: A child's dominant parent must supply at least this share of
#: the inflow to keep the edge on the main stem.  0.5 is a strict majority;
#: lower it toward plurality to keep co-fed branches.  Sweep it, like `thr.s`.
MAIN_FEEDER_FRAC = 0.5


# ══════════════════════════════════════════════════════════════════════════════
def edge_fluxes(Lp: np.ndarray, Lc: np.ndarray, U: np.ndarray,
                V: np.ndarray) -> dict[tuple[int, int], float]:
    """{(parent, child): inward IVT flux from parent into child}; parent 0 = ambient.

    Ambient (label 0 just outside the child) is reported so the diagnostic
    `ambient_share_tc` can say how environmentally fed the main stem is, but it
    can never own a parcel.
    """
    out: dict[tuple[int, int], float] = defaultdict(float)
    children = [int(c) for c in np.unique(Lc) if c > 0]
    if not children:
        return dict(out)
    d1 = disk(1)
    for c in children:
        mc = Lc == c
        er = ndi.binary_erosion(mc, ST, border_value=0)
        bnd = mc & ~er                                # inner boundary of the child
        if not bnd.any():
            continue
        gy, gx = np.gradient(mc.astype(np.float32))   # points inward at the boundary
        infl = np.clip(U * gx + V * gy, 0.0, None)    # inward flux, positive part
        ext = ndi.binary_dilation(mc, ST) & ~mc
        p_at_ext = np.where(ext, Lp, 0).astype(np.int32)
        p_face = ndi.grey_dilation(p_at_ext, footprint=d1)
        b = np.nonzero(bnd)
        for y, x, fv in zip(b[0], b[1], infl[b]):
            if fv > 0:
                out[(int(p_face[y, x]), c)] += float(fv)
    return dict(out)


def component_measure(lab2d: np.ndarray, field2d: np.ndarray, area: np.ndarray,
                      measure: str) -> dict[int, float]:
    """Per-object scalar for the content / peak / mean measures.

    `lab2d` is whichever granularity the sweep is running at: components for the
    `stem` variant, basins for `stem_ws`.  The measure does not care, which is
    the point of separating it.
    """
    out: dict[int, float] = {}
    fa = field2d * area if measure == "content" else None
    for c in np.unique(lab2d):
        c = int(c)
        if c == 0:
            continue
        m = lab2d == c
        if not m.any():
            continue
        if measure == "content":
            out[c] = float(fa[m].sum())               # integrated column water, kg
        elif measure == "peak":
            out[c] = float(field2d[m].max())
        elif measure == "mean":
            out[c] = float(field2d[m].mean())
        else:
            raise ValueError(f"component_measure does not handle {measure!r}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
def resolve_owner(contributions: dict[int, float], frac: float,
                  ambient: float = 0.0, *, include_ambient: bool = False
                  ) -> tuple[int | None, float, float]:
    """The dominant lineage, or None when no one clears `frac`.

    Returns (winner or None, winner's share, denominator).  `include_ambient`
    puts the environmental inflow in the denominator, which only the flux
    measure has; for the mass measures the denominator is the detected parents
    alone

    A single contributor always wins regardless of `frac`: one parent is a
    CONTINUATION, not a merge, and a lone remnant coasting on ambient inflow
    must keep its identity rather than be orphaned every frame.
    """
    if not contributions:
        return None, 0.0, 0.0
    if len(contributions) == 1:
        only = next(iter(contributions))
        return only, 1.0, float(contributions[only])
    real = float(sum(contributions.values()))
    den = real + (float(ambient) if include_ambient else 0.0)
    if den <= 0:
        return None, 0.0, 0.0
    best = max(contributions, key=contributions.get)
    share = contributions[best] / den
    return (best if share >= frac else None), float(share), den


def describe(measure: str) -> str:
    """One-line statement of what the measure claims, for a figure caption."""
    return {
        "content": "integrated column water vapour at the merge (mass: who brought "
                   "more water)",
        "peak": "peak column water vapour at the merge (intensity, not mass)",
        "mean": "mean column water vapour at the merge (intensity, not mass)",
        "flux": "directed boundary IVT flux (instantaneous transport share, not a "
                "closed budget)",
    }[measure]


def needs_field(measure: str) -> str | None:
    """Which stored field the measure reads: 'iwv', 'uv', or None."""
    if measure == "flux":
        return "uv"
    if measure in ("content", "peak", "mean"):
        return "iwv"
    return None
