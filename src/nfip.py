"""NFIP claims -- loading, and the space-time prefilter that scopes a storm.

THE PREFILTER IS LOOSE ON PURPOSE
    2000 km and +/-72 h.  That is not an attribution radius; it is the envelope
    of claims a storm could conceivably be associated with, and the corridor
    decides which of them actually are.  A tight prefilter would make the
    corridor's answer a function of the prefilter, which is the failure it
    exists to avoid.

    The 72 h matches `time.tail_days = 3` exactly, and it must: if the claim
    window outran the corridor's frame axis, the extra claims could only ever
    be recorded as `out_of_window`, inflating the denominator without ever
    being classifiable.  `tcmc.hazards.registry` gives the NFIP class
    `tail_days = 3` for that reason and `Config.validate` checks the pair.

dateOfLoss IS DATE-ONLY AND MISREPORTED
    which is why the matching tolerance is +/-24 h and is labelled MEASUREMENT
    ERROR rather than physical lag.  A tornado report carries minute-resolution
    `begin_time` and gets +/-3 h, half the frame spacing, which is a field
    resolution allowance.  The two numbers look similar and mean different
    things; the registry keeps them apart.  Sweep the claim tolerance over
    {12h, 24h, 48h}.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tcmc.geo import haversine_km

#: Column names in the clustered-claims CSV.  Overridable per call, because a
#: differently prepared CSV should be a keyword argument rather than a fork.
DEFAULT_COLS = dict(lat="latitude", lon="longitude", date="dateOfLoss")


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in (
        ("uid", "string"), ("hazard", "string"), ("lat", "float64"),
        ("lon", "float64"), ("time", "datetime64[ns]"),
        ("cluster_id", "string"), ("distance_km", "float64"),
        ("time_delta_hours", "float64"))})


# ══════════════════════════════════════════════════════════════════════════════
def load_claims(cfg, *, path: str | None = None, verbose: bool = True
                ) -> pd.DataFrame | None:
    """Read the clustered claims CSV once.  Returns None when unavailable.

    None rather than raising: a missing claims file makes the NFIP hazard
    unavailable on this machine, which the stage records per storm, and should
    not stop a convective-only run.  `Config.check_inputs` reports it.
    """
    import os
    p = path or cfg.hazards.claims_csv
    if not p or not os.path.exists(p):
        if verbose:
            print(f"NFIP claims unavailable ({p}); the nfip hazard will be "
                  "recorded as unavailable rather than as zero")
        return None

    df = pd.read_csv(p, low_memory=False)
    c = DEFAULT_COLS
    out = pd.DataFrame({
        "lat": pd.to_numeric(df[c["lat"]], errors="coerce"),
        "lon": pd.to_numeric(df[c["lon"]], errors="coerce"),
        "time": pd.to_datetime(df[c["date"]], errors="coerce"),
        "cluster_id": df[cfg.hazards.claims_cluster_col].astype("string"),
    }).dropna(subset=["lat", "lon", "time", "cluster_id"]).reset_index(drop=True)

    # Everything downstream is tz-naive -- track times, Storm Events begin_time
    # and the frame axis all are -- and a tz-aware column compares unequal.
    try:
        if getattr(out["time"].dt, "tz", None) is not None:
            out["time"] = out["time"].dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass

    out["hazard"] = "nfip"
    out["uid"] = [claim_uid(r.lat, r.lon, r.time, r.cluster_id)
                  for r in out.itertuples()]
    if verbose:
        print(f"NFIP claims: {len(out):,} "
              f"({out['cluster_id'].nunique():,} clusters, "
              f"{out['time'].dt.year.min():.0f}-{out['time'].dt.year.max():.0f})")
    return out


def claim_uid(lat, lon, date, cluster_id=None) -> str:
    """Stable id for a claim.

    NFIP has no natural key, so the identifying tuple is hashed; identical
    duplicates collapse, which is the desired behaviour.  Stable across runs,
    so a link table can be joined back to the source record.
    """
    import hashlib
    s = "|".join(str(x) for x in (round(float(lat), 5), round(float(lon), 5),
                                  pd.Timestamp(date).strftime("%Y-%m-%d"),
                                  cluster_id))
    return hashlib.blake2b(s.encode(), digest_size=8).hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
def points_for_track(claims: pd.DataFrame | None, track: pd.DataFrame, cfg,
                     *, chunk: int = 20_000) -> pd.DataFrame:
    """Claims whose (space, time) intersects a track fix.

    A claim is kept iff SOME fix is within `claim_r_km` AND within
    +/-`claim_t_hours` of it.  For each kept claim the distance and signed time
    offset to its single CLOSEST fix are also reported, which are metadata for
    later inspection and play no part in the keep decision.

    Chunked over claims to bound the claim x fix matrix: at 2000 km and a long
    track the naive outer product is tens of gigabytes, and the temporal and
    bounding-box prefilters below already remove most of it.
    """
    if claims is None or not len(claims) or not len(track):
        return _empty()

    r_km = cfg.hazards.claim_r_km
    t_hours = cfg.hazards.claim_t_hours
    speed_kmh = 30.0        # only ranks the closest fix; never gates the keep

    fix_t = pd.to_datetime(track["iso_time"]).to_numpy("datetime64[ns]")
    fix_lat = track["lat"].to_numpy(float)
    fix_lon = track["lon"].to_numpy(float)

    pad = pd.Timedelta(hours=t_hours)
    tmin = pd.Timestamp(fix_t.min()) - pad
    tmax = pd.Timestamp(fix_t.max()) + pad
    c = claims[(claims["time"] >= tmin) & (claims["time"] <= tmax)]

    pad_deg = r_km / 111.0
    c = c[c["lat"].between(fix_lat.min() - pad_deg, fix_lat.max() + pad_deg)
          & c["lon"].between(fix_lon.min() - pad_deg, fix_lon.max() + pad_deg)]
    if c.empty:
        return _empty()
    c = c.reset_index(drop=True)

    c_t = c["time"].to_numpy("datetime64[ns]")
    c_lat = c["lat"].to_numpy(float)
    c_lon = c["lon"].to_numpy(float)
    keep = np.zeros(len(c), bool)
    best_d = np.full(len(c), np.nan)
    best_dt = np.full(len(c), np.nan)

    for s in range(0, len(c), chunk):
        e = min(s + chunk, len(c))
        dt_h = (c_t[s:e, None] - fix_t[None, :]) / np.timedelta64(1, "h")
        dist = haversine_km(c_lat[s:e, None], c_lon[s:e, None],
                            fix_lat[None, :], fix_lon[None, :])
        within = (np.abs(dt_h) <= t_hours) & (dist <= r_km)
        blk = within.any(axis=1)
        keep[s:e] = blk
        score = np.where(within, dist + np.abs(dt_h) * speed_kmh, np.inf)
        j = score.argmin(axis=1)
        rows = np.arange(e - s)
        best_d[s:e][blk] = dist[rows[blk], j[blk]]
        best_dt[s:e][blk] = dt_h[rows[blk], j[blk]]

    out = c.loc[keep, ["uid", "hazard", "lat", "lon", "time", "cluster_id"]].copy()
    out["distance_km"] = best_d[keep]
    out["time_delta_hours"] = best_dt[keep]
    return out.sort_values("time").reset_index(drop=True)
