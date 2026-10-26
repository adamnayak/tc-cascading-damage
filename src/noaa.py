"""NOAA Storm Events -- one national table, built once, every hazard class.

WHAT CHANGED
    The previous `events_for_track` opened and parsed a whole year's details
    CSV for every storm.  At ~500 storms spanning one or two years each, that
    is ~800 parses of files averaging several hundred thousand rows, and it was
    the dominant cost of an otherwise instant stage.  Here the yearly CSVs are
    parsed ONCE into a filtered national parquet, and each storm slices it.

    Adding a hazard class is then free at the network level: the yearly
    gzipped CSVs are cached whole and unfiltered, so Flash Flood, Flood, Heavy
    Rain, Hail and Lightning require no new downloads -- only a wider type
    filter and one rebuild of the national table.  That was the main finding of
    the scoping work and it is why the type list comes from
    `tcmc.hazards.registry` rather than a module constant.

COVERAGE
    Tornado from 1950, thunderstorm wind and hail from 1955, everything else
    from 1996.  A zero count for `flood` in a 1985 storm is not zero, it is
    unobservable, and the attribution table records `coverage=False` for those
    rows.  The registry owns those start years; nothing here hardcodes them.

SOURCE
    The AWS Open Data mirror (NODD) is tried first and the NCEI HTTPS
    autoindex is the fallback, because scraping the directory listing is
    fragile.  Both the file index and every yearly archive are cached on disk,
    so a rebuild after adding a hazard class never re-hits the network.
"""
from __future__ import annotations

import io
import json
import os
import re

import numpy as np
import pandas as pd

from tcmc.geo import CONUS_BBOX
from tcmc.hazards import registry as HR

NCEI_MIRRORS = [
    "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/",
    "https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/",
]
S3_BUCKET = "noaa-swdi-pds"
S3_PREFIX_CANDIDATES = ["stormevents/csvfiles/", "swdi/stormevents/csvfiles/",
                        "csvfiles/"]
DEFAULT_TIMEOUT = (10, 120)
_DETAILS_RE = re.compile(r"StormEvents_details-ftp_v1\.0_d(\d{4})_c\d+\.csv\.gz")
START_YEAR = 1950

CONUS_STATES = {
    "ALABAMA", "ARIZONA", "ARKANSAS", "CALIFORNIA", "COLORADO", "CONNECTICUT",
    "DELAWARE", "DISTRICT OF COLUMBIA", "FLORIDA", "GEORGIA", "IDAHO", "ILLINOIS",
    "INDIANA", "IOWA", "KANSAS", "KENTUCKY", "LOUISIANA", "MAINE", "MARYLAND",
    "MASSACHUSETTS", "MICHIGAN", "MINNESOTA", "MISSISSIPPI", "MISSOURI", "MONTANA",
    "NEBRASKA", "NEVADA", "NEW HAMPSHIRE", "NEW JERSEY", "NEW MEXICO", "NEW YORK",
    "NORTH CAROLINA", "NORTH DAKOTA", "OHIO", "OKLAHOMA", "OREGON", "PENNSYLVANIA",
    "RHODE ISLAND", "SOUTH CAROLINA", "SOUTH DAKOTA", "TENNESSEE", "TEXAS", "UTAH",
    "VERMONT", "VIRGINIA", "WASHINGTON", "WEST VIRGINIA", "WISCONSIN", "WYOMING",
}


# ══════════════════════════════════════════════════════════════════════════════
# HTTP
# ══════════════════════════════════════════════════════════════════════════════
def make_session(total_retries: int = 5, backoff: float = 0.6):
    """A requests.Session that retries transient failures with backoff."""
    import requests
    from requests.adapters import HTTPAdapter
    try:
        from urllib3.util.retry import Retry
    except ImportError:                                        # pragma: no cover
        from requests.packages.urllib3.util.retry import Retry
    retry = Retry(total=total_retries, connect=total_retries, read=total_retries,
                  backoff_factor=backoff,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset({"GET", "HEAD"}), raise_on_status=False)
    s = requests.Session()
    ad = HTTPAdapter(max_retries=retry)
    s.mount("https://", ad)
    s.mount("http://", ad)
    s.headers.update({"User-Agent": "tcmc/2.0"})
    return s


def _s3_index() -> dict:
    try:
        import s3fs
    except ImportError:
        return {}
    fs = s3fs.S3FileSystem(anon=True)
    for prefix in S3_PREFIX_CANDIDATES:
        try:
            hits = fs.glob(f"{S3_BUCKET}/{prefix}"
                           "StormEvents_details-ftp_v1.0_d*.csv.gz")
        except Exception:                                      # noqa: BLE001
            hits = []
        idx = {}
        for key in hits or []:
            m = _DETAILS_RE.search(key)
            if m:
                idx[int(m.group(1))] = "s3://" + key.lstrip("/")
        if idx:
            return dict(sorted(idx.items()))
    return {}


def _ncei_index(session) -> dict:
    for base in NCEI_MIRRORS:
        try:
            html = session.get(base, timeout=DEFAULT_TIMEOUT).text
        except Exception:                                      # noqa: BLE001
            continue
        idx = {}
        for full in re.findall(
                r"StormEvents_details-ftp_v1\.0_d\d{4}_c\d+\.csv\.gz", html):
            idx[int(_DETAILS_RE.search(full).group(1))] = base + full
        if idx:
            return dict(sorted(idx.items()))
    return {}


def file_index(cache_dir: str = "noaa_cache", *, session=None,
               refresh: bool = False) -> dict:
    """{year: url}, cached to JSON so repeated runs skip discovery."""
    path = os.path.join(cache_dir, "details_index.json")
    if not refresh and os.path.exists(path):
        with open(path) as fh:
            return {int(k): v for k, v in json.load(fh).items()}
    session = session or make_session()
    idx = _s3_index() or _ncei_index(session)
    if not idx:
        raise RuntimeError("could not resolve the Storm Events file index from "
                           "S3 or NCEI")
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(idx, fh)
    return idx


def _fetch(url: str, session) -> bytes:
    if url.startswith("s3://"):
        import fsspec
        with fsspec.open(url, "rb", anon=True) as fh:
            return fh.read()
    return session.get(url, timeout=DEFAULT_TIMEOUT).content


def year_csv(year: int, index: dict, cache_dir: str, session) -> pd.DataFrame:
    """One year's details CSV, downloaded once and cached whole.

    Cached UNFILTERED on purpose: widening the hazard set must not require a
    re-download, and the gzipped archive is smaller than the filtered parquet
    would be for a broad selection anyway.
    """
    if year not in index:
        raise ValueError(f"no Storm Events details file for {year}")
    url = index[year]
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, os.path.basename(url))
    if not os.path.exists(path):
        with open(path, "wb") as fh:
            fh.write(_fetch(url, session))
    return pd.read_csv(path, compression="gzip", low_memory=False)


# ══════════════════════════════════════════════════════════════════════════════
# Parsing
# ══════════════════════════════════════════════════════════════════════════════
def _times(df: pd.DataFrame) -> pd.Series:
    ym = df["BEGIN_YEARMONTH"].astype("Int64").astype(str).str.zfill(6)
    day = df["BEGIN_DAY"].fillna(1).astype(int).astype(str).str.zfill(2)
    tm = df["BEGIN_TIME"].fillna(0).astype(int).astype(str).str.zfill(4)
    return pd.to_datetime(ym + day + tm, format="%Y%m%d%H%M", errors="coerce")


def parse_year(raw: pd.DataFrame, hazard_keys, *, conus: bool = True
               ) -> pd.DataFrame:
    """One year's raw CSV -> tidy hazard points for the enabled classes."""
    types = HR.noaa_event_types(hazard_keys)
    haz_of = HR.hazard_of_event_type(hazard_keys)
    if not types:
        return _empty()

    df = raw[raw["EVENT_TYPE"].isin(types)]
    if not len(df):
        return _empty()
    if conus and "STATE" in df.columns:
        df = df[df["STATE"].astype(str).str.upper().isin(CONUS_STATES)]

    out = pd.DataFrame({
        "uid": df.get("EVENT_ID", pd.Series(index=df.index)).astype("string"),
        "hazard": df["EVENT_TYPE"].map(haz_of).astype("string"),
        "event_type": df["EVENT_TYPE"].astype("string"),
        "state": df.get("STATE", pd.Series(index=df.index)).astype("string"),
        "time": _times(df),
        "lat": pd.to_numeric(df.get("BEGIN_LAT"), errors="coerce"),
        "lon": pd.to_numeric(df.get("BEGIN_LON"), errors="coerce"),
        "magnitude": pd.to_numeric(df.get("MAGNITUDE"), errors="coerce"),
        "tor_f_scale": df.get("TOR_F_SCALE",
                              pd.Series(index=df.index)).astype("string"),
    })
    # Rows without coordinates cannot be classified against a corridor at all.
    # Common before 1996 and reported by `build_table` rather than hidden.
    out = out.dropna(subset=["time", "lat", "lon"])
    b = CONUS_BBOX
    out = out[out["lon"].between(b["lon_min"], b["lon_max"])
              & out["lat"].between(b["lat_min"], b["lat_max"])]
    return out.reset_index(drop=True)


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in (
        ("uid", "string"), ("hazard", "string"), ("event_type", "string"),
        ("state", "string"), ("time", "datetime64[ns]"), ("lat", "float64"),
        ("lon", "float64"), ("magnitude", "float64"),
        ("tor_f_scale", "string"))})


# ══════════════════════════════════════════════════════════════════════════════
# The national table
# ══════════════════════════════════════════════════════════════════════════════
def table_path(cfg) -> str:
    """One file per hazard selection, so changing it cannot read a stale table."""
    import hashlib
    key = ",".join(sorted(h for h in cfg.hazards.enabled
                          if HR.get(h).source == "noaa"))
    tag = hashlib.blake2b(key.encode(), digest_size=4).hexdigest()
    return os.path.join(cfg.hazards.noaa_cache,
                        f"events_{cfg.population.year_start}_"
                        f"{cfg.population.year_end}_{tag}.parquet")


def build_table(cfg, *, session=None, refresh: bool = False,
                verbose: bool = True) -> pd.DataFrame:
    """Build (or read) the national hazard-point table for this configuration.

    Reports the dropped-for-missing-coordinates count per decade, because that
    is a real and strongly time-varying limitation: older years often lack
    BEGIN_LAT/BEGIN_LON entirely, and a report with no position cannot be
    attributed to a corridor by any method.
    """
    path = table_path(cfg)
    if os.path.exists(path) and not refresh:
        return pd.read_parquet(path)

    session = session or make_session()
    idx = file_index(cfg.hazards.noaa_cache, session=session)
    y0 = max(cfg.population.year_start, START_YEAR)
    y1 = cfg.population.year_end
    # The tail can push a late-December storm's window into the next year.
    y1_read = y1 + 1

    frames, dropped = [], []
    for y in range(y0, y1_read + 1):
        if y not in idx:
            continue
        raw = year_csv(y, idx, cfg.hazards.noaa_cache, session)
        n_raw = int(raw["EVENT_TYPE"].isin(
            HR.noaa_event_types(cfg.hazards.enabled)).sum())
        ev = parse_year(raw, cfg.hazards.enabled, conus=cfg.hazards.conus_only)
        dropped.append(dict(year=y, matched=n_raw, kept=len(ev),
                            dropped=n_raw - len(ev)))
        if len(ev):
            frames.append(ev)
        del raw
        if verbose and y % 10 == 0:
            print(f"  ... {y}")

    out = pd.concat(frames, ignore_index=True) if frames else _empty()
    out = out.sort_values("time").reset_index(drop=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out.to_parquet(path, index=False)

    if verbose:
        d = pd.DataFrame(dropped)
        print(f"Storm Events table: {len(out):,} points, "
              f"{out['hazard'].nunique()} hazard classes, {y0}-{y1_read}")
        if len(d) and d["dropped"].sum():
            d["decade"] = (d["year"] // 10) * 10
            g = d.groupby("decade")[["matched", "dropped"]].sum()
            g["pct_dropped"] = (100 * g["dropped"] / g["matched"].clip(lower=1)).round(1)
            print("  dropped for missing coordinates (unattributable by ANY method):")
            print(g.to_string())
        print("  " + HR.coverage_note(cfg.hazards.enabled,
                                      cfg.population.year_start,
                                      cfg.population.year_end))
    return out


def load_table(cfg) -> pd.DataFrame:
    path = table_path(cfg)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not built. Call tcmc.hazards.noaa.build_table(cfg) once "
            "per hazard selection; it parses the cached yearly CSVs and needs "
            "no network if they are already downloaded.")
    return pd.read_parquet(path)


# ══════════════════════════════════════════════════════════════════════════════
def points_for_window(table: pd.DataFrame, start, end) -> pd.DataFrame:
    """Hazard points inside a time window.  A slice, not a re-parse."""
    if table is None or not len(table):
        return _empty()
    t = table["time"]
    return table[(t >= pd.Timestamp(start)) & (t <= pd.Timestamp(end))] \
        .reset_index(drop=True)


def points_for_track(table: pd.DataFrame, track: pd.DataFrame, cfg,
                     *, bbox: dict | None = None) -> pd.DataFrame:
    """Hazard points in this storm's space-time window.

    The temporal window is the FRAME AXIS, not the track: a report after the
    last corridor frame cannot be classified, and fetching it only to record it
    as `out_of_window` inflates every denominator.  `Config.effective_tail_days`
    therefore governs the fetch and the corridor identically.
    """
    t = pd.to_datetime(track["iso_time"])
    start = t.min() - pd.Timedelta(days=cfg.time.lead_days)
    end = t.max() + pd.Timedelta(days=cfg.fetch_tail_days)
    out = points_for_window(table, start, end)
    if bbox is not None and len(out):
        out = out[out["lon"].between(bbox["lon_min"], bbox["lon_max"])
                  & out["lat"].between(bbox["lat_min"], bbox["lat_max"])]
    return out.reset_index(drop=True)
