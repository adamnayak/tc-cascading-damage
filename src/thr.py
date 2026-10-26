"""Stage A -- THR detection.
Reference works:
    Vincent (1993), IEEE TIP 2(2):176-201  -- morphological reconstruction
    Xu, Ma, Chang & Wang (2020), GMD 13:4639-4662 (IPART), Sect. 2.1 / App. A1
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import reconstruction as _reconstruct

from tcmc.geo import cell_area_km2  # noqa: F401  (re-exported; callers expect it here)

#: IPART thr.py: "reconstruction element: a 6-connectivity element".
_CROSS2 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
_CROSS3 = np.zeros((3, 3, 3), bool)
_CROSS3[0, 1, 1] = _CROSS3[2, 1, 1] = True
_CROSS3[1, :, 1] = _CROSS3[1, 1, :] = True

#: 4-connectivity, used for every labelling and morphology step in the corridor.
ST = ndi.generate_binary_structure(2, 1)


def disk(r: int) -> np.ndarray:
    """Binary disk of radius `r` cells; a 1x1 element for r <= 0."""
    if r <= 0:
        return np.ones((1, 1), bool)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (y * y + x * x) <= r * r


# ══════════════════════════════════════════════════════════════════════════════
def anomaly(f: np.ndarray, s: int = 18, t: int = 0) -> np.ndarray:
    """Top-hat by reconstruction.  anomaly = f - reconstruction(erosion(f, E), f).

    `f` is (T, Y, X) float |IVT|.  Returns anomaly >= 0, float32, same shape.
    About 6 ms per frame at 160x220.

    `t = 0` makes the reconstruction per-frame and two-dimensional, which is the
    setting used throughout: IPART's default temporal half-width is 4 days, and
    blurring across four days would blur tau, the quantity being measured.
    """
    f = np.ascontiguousarray(f, np.float64)          # reconstruction needs float
    size = (2 * int(t) + 1, 2 * int(s) + 1, 2 * int(s) + 1)
    marker = ndi.minimum_filter(f, size=size, mode="nearest")   # greyscale erosion
    if t == 0:
        rec = np.stack([_reconstruct(marker[k], f[k], "dilation", footprint=_CROSS2)
                        for k in range(f.shape[0])])
    else:
        rec = _reconstruct(marker, f, "dilation", footprint=_CROSS3)
    return np.asarray(f - rec, np.float32)


def detect(anom: np.ndarray, area_km2: np.ndarray, *, floor: float = 45.0,
           min_area_km2: float = 2.5e4, open_radius: int = 1) -> np.ndarray:
    """anomaly -> candidate mask.  Threshold, open, drop small components.

    `floor` is IPART's `thres_low`: a constant on an ANOMALY, after the
    background has been removed, so it transfers across regimes.

    `min_area_km2` is OURS.  IPART's 5e5 km^2 is an AR length filter in disguise
    (an AR must be >= 2000 km long), and a frontal band (~165 x 1450 km ~
    2.4e5 km^2) or a secondary plume falls below it and is SILENTLY DELETED --
    which removes exactly the intersecting systems this pipeline exists to count.
    2.5e4 km^2 is about 30 cells at 0.25 deg, i.e. "a moisture system is at least
    ~150 km across".

    Opening severs sub-resolution bridges.  Binary connectivity is transitive, so
    one supra-threshold cell touching an unrelated front merges the whole front.
    """
    M = anom > float(floor)
    out = np.zeros_like(M)
    st = disk(open_radius)
    for k in range(M.shape[0]):
        m = M[k]
        if open_radius > 0 and m.any():
            m = ndi.binary_opening(m, structure=st)
        if not m.any():
            continue
        if min_area_km2 > 0:
            lab, n = ndi.label(m, structure=ST)
            if n:
                areas = ndi.sum_labels(area_km2, lab, index=np.arange(1, n + 1))
                keep = 1 + np.flatnonzero(areas >= min_area_km2)
                m = np.isin(lab, keep) if keep.size else np.zeros_like(m)
        out[k] = m
    return out


def components(mask: np.ndarray) -> np.ndarray:
    """Per-frame connected components, 4-connectivity.  int32, 0 = background.

    Frame-local labels.  Temporally persistent identity is the `basin` field
    produced by `tcmc.corridor.segment`, not this.
    """
    return np.stack([ndi.label(mask[k], structure=ST)[0].astype(np.int32)
                     for k in range(mask.shape[0])])


def detect_all(ivt: np.ndarray, lat: np.ndarray, lon: np.ndarray, *,
               s: int = 18, t: int = 0, floor: float = 45.0,
               min_area_km2: float = 2.5e4, open_radius: int = 1) -> dict:
    """|IVT| -> {anomaly, mask, component, area_km2}.  The whole of detection.

    One call so the segment stage, the notebook and the diagnostics cannot drift
    apart on parameter order.
    """
    A = cell_area_km2(lat, lon)
    an = anomaly(np.asarray(ivt, np.float32), s=s, t=t)
    m = detect(an, A, floor=floor, min_area_km2=min_area_km2,
               open_radius=open_radius)
    return dict(anomaly=an, mask=m, component=components(m), area_km2=A)
