"""Stage B -- track geometry and seeds.

The track is a MARKER, not a spine.  A corridor cell is TC-owned because it
contains a marker or because it continues something that did, never because it
lies near the track line.  Frames after the last fix keep the last centre with
`in_track=False`: that is the remnant, where the corridor must survive without a
track and where most of the late attribution happens.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tcmc.geo import epoch_s, haversine_km, unwrap_lon, wrap_lon180


def track_centers(track: pd.DataFrame, lon: np.ndarray, lat: np.ndarray,
                  frame_times) -> dict:
    """Per-frame centre in grid indices and degrees, plus the in-track flag.

    Longitude is unwrapped before interpolation and re-wrapped after, so a track
    crossing the dateline does not interpolate through zero.  (The original
    `corridor_THR.track_centers` interpolated raw longitude; harmless for the
    North Atlantic, wrong for any other basin, and free to fix.)
    """
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    ft = epoch_s(frame_times)
    tt = epoch_s(track["iso_time"])

    in_track = (ft >= tt.min()) & (ft <= tt.max())
    flat = np.interp(ft, tt, track["lat"].to_numpy(float))
    flon = wrap_lon180(np.interp(ft, tt, unwrap_lon(track["lon"].to_numpy(float))))

    ii = np.abs(lat[None, :] - flat[:, None]).argmin(1)
    jj = np.abs(lon[None, :] - flon[:, None]).argmin(1)
    return dict(i=ii, j=jj, in_track=in_track, lat=flat, lon=flon)


def marker_mask(track: pd.DataFrame, lon: np.ndarray, lat: np.ndarray,
                frame_times, radius_km: float = 250.0) -> np.ndarray:
    """(T, Y, X) bool: cells within `radius_km` of the centre, while a track exists.

    Re-seeded every in-track frame, so a corridor that fragments and re-forms
    around the centre is recovered rather than lost.
    """
    c = track_centers(track, lon, lat, frame_times)
    Lon, Lat = np.meshgrid(np.asarray(lon, float), np.asarray(lat, float))
    T = len(np.asarray(frame_times))
    M = np.zeros((T, Lat.shape[0], Lat.shape[1]), bool)
    for k in range(T):
        if c["in_track"][k]:
            M[k] = haversine_km(c["lat"][k], c["lon"][k], Lat, Lon) <= radius_km
    return M


def courant(mask: np.ndarray, ii: np.ndarray, jj: np.ndarray) -> np.ndarray:
    """C_link = |centre displacement| / (mask extent along that displacement).

    Overlap linking succeeds iff C_link < 1.  Frames at or above 1 are frames
    where the object may have moved further than its own along-track extent and
    the graph can fracture, which is the diagnostic that decides whether 6 h is
    enough -- pay for 3 h only where it fires.

    This matters MORE after watershedding, because basins are smaller than the
    components they subdivide, so the same displacement is a larger fraction of
    the object.  Computed per variant for exactly that reason.  If it fires
    often, the principled fix is a semi-Lagrangian frame built from the stored
    IVT vectors (Jones et al. 2023, AMT 16:1043), not a finer cadence.
    """
    T = mask.shape[0]
    out = np.full(T, np.nan)
    for k in range(T - 1):
        di, dj = ii[k + 1] - ii[k], jj[k + 1] - jj[k]
        D = float(np.hypot(di, dj))
        m = mask[k]
        if not m.any():
            continue
        if D == 0:
            out[k] = 0.0
            continue
        y, x = np.nonzero(m)
        proj = (y * di + x * dj) / D
        out[k] = D / max(float(proj.max() - proj.min()) + 1.0, 1e-9)
    return out


def edge_stats(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, *,
               hazard_bbox: dict | None = None) -> dict:
    """How often, and where, a mask touches the domain edge.

    Carried over from `domain_v2.edge_stats`.  The point is to DEMONSTRATE that
    the corridor is not leaving the domain over land rather than assert it: an
    exit over the Maritimes or open ocean costs no attributable hazard, an exit
    into the Ohio Valley costs the densest claim region in the record.

    North and east edges are resolved from coordinate VALUES, not from index 0
    or -1, because latitude is stored descending.
    """
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    m = np.asarray(field, bool)
    if m.ndim == 2:
        m = m[None, :, :]
    T = m.shape[0]

    i_north = int(np.argmax(lat))
    i_south = int(np.argmin(lat))
    j_east = int(np.argmax(lon))
    j_west = int(np.argmin(lon))

    touch = {}
    for side, sl in (("north", (slice(None), i_north, slice(None))),
                     ("south", (slice(None), i_south, slice(None))),
                     ("east", (slice(None), slice(None), j_east)),
                     ("west", (slice(None), slice(None), j_west))):
        touch[side] = m[sl].reshape(T, -1).any(axis=1)

    any_edge = np.zeros(T, bool)
    for v in touch.values():
        any_edge |= v

    in_hazard = np.zeros(T, bool)
    if hazard_bbox:
        b = hazard_bbox
        rows = (lat >= b["lat_min"]) & (lat <= b["lat_max"])
        cols = (lon >= b["lon_min"]) & (lon <= b["lon_max"])
        for side, (ii, jj) in (("north", (i_north, None)), ("south", (i_south, None)),
                               ("east", (None, j_east)), ("west", (None, j_west))):
            if ii is not None and rows[ii]:
                in_hazard |= (m[:, ii, :][:, cols]).any(axis=1)
            if jj is not None and cols[jj]:
                in_hazard |= (m[:, :, jj][:, rows]).any(axis=1)

    return dict(
        edge_frames=int(any_edge.sum()),
        edge_frames_in_hazard_bbox=int(in_hazard.sum()),
        **{f"edge_{s}_frames": int(v.sum()) for s, v in touch.items()},
    )
