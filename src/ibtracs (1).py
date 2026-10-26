"""IBTrACS ingestion -- a drop-in alternative to `hurdat`, same tidy schema.

Default source is the North Atlantic basin CSV of IBTrACS v04r01 (Knapp et al.,
2010).  The full storm lifecycle is kept (no NATURE filter), matching the
HURDAT2 population.  Positions use LAT/LON (-180..180, deg N/E); winds and
pressures prefer the USA agency columns and fall back to WMO.

THE NA TRAP
    `keep_default_na=False` is not optional.  The basin code for the North
    Atlantic is the string "NA", and pandas' default missing-value tokens
    include "NA", so the basin filter silently matches nothing and the parser
    returns an empty frame.  Explicit `na_values` are supplied instead.

SIZE COLUMNS
    `load_size` pulls USA_RMW and the R34 quadrant radii, which the parametric
    baseline needs for rmax.  They are read from the same CSV rather than a
    second download, but kept out of `parse` so the corridor pipeline does not
    carry columns it never uses.
"""
from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd

DEFAULT_URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.NA.list.v04r01.csv"
)
ATLANTIC_BASIN = "NA"
ERA5_START_YEAR = 1940
SATELLITE_ERA_YEAR = 1979

_NA_TOKENS = ("", " ", "MM", "NA", "NOT_NAMED", "UNNAMED")
_READ_KW = dict(skiprows=[1], low_memory=False, keep_default_na=False,
                na_values=[" ", "", "MM", "-999", "-1073741824"])

#: Quadrant radii of 34 kt wind, nautical miles.  Used by the parametric baseline.
R34_COLS = ("USA_R34_NE", "USA_R34_SE", "USA_R34_SW", "USA_R34_NW")


def _num(series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def read_csv(text_or_path: str) -> pd.DataFrame:
    """Raw IBTrACS frame, with the units row skipped and "NA" preserved."""
    if os.path.exists(text_or_path):
        return pd.read_csv(text_or_path, **_READ_KW)
    return pd.read_csv(io.StringIO(text_or_path), **_READ_KW)


def parse(text_or_df, basin: str = ATLANTIC_BASIN) -> pd.DataFrame:
    """Parse an IBTrACS CSV (text, path, or preloaded frame) into the tidy schema."""
    df = text_or_df.copy() if isinstance(text_or_df, pd.DataFrame) \
        else read_csv(text_or_df)

    if "BASIN" in df.columns and basin:
        df = df[df["BASIN"].astype(str).str.upper() == basin.upper()]

    lon = ((_num(df["LON"]) + 180) % 360) - 180

    name = df["NAME"].astype(str).str.upper().str.strip()
    name = name.where(~name.isin([t.upper() for t in _NA_TOKENS]), "UNNAMED")

    wind = _num(df.get("USA_WIND"))
    if "WMO_WIND" in df.columns:
        wind = wind.fillna(_num(df["WMO_WIND"]))
    pres = _num(df.get("USA_PRES"))
    if "WMO_PRES" in df.columns:
        pres = pres.fillna(_num(df["WMO_PRES"]))

    # USA_ATCF_ID ("AL092017") is the id the hazard and damage figures key on,
    # and it is NOT the IBTrACS SID ("2017228N14314").  Carried through so the
    # analysis notebooks can join on it without re-reading the source CSV.
    atcf = (df["USA_ATCF_ID"].astype(str).str.strip().str.upper()
            if "USA_ATCF_ID" in df.columns else pd.Series("", index=df.index))
    atcf = atcf.where(~atcf.isin(["", "NAN", "NA", "MM"]), pd.NA)

    out = pd.DataFrame({
        "storm_id": df["SID"].astype(str),
        "atcf": atcf,
        "storm_name": name,
        "iso_time": pd.to_datetime(df["ISO_TIME"], errors="coerce"),
        "year": _num(df["SEASON"]),
        "lat": _num(df["LAT"]),
        "lon": lon,
        "status": df["NATURE"].astype(str) if "NATURE" in df.columns else "",
        "max_wind_kt": wind,
        "min_mslp_mb": pres,
    })
    out = out.dropna(subset=["iso_time", "lat", "lon", "year"]).reset_index(drop=True)
    out["year"] = out["year"].astype(int)
    return out


def load_size(text_or_df, basin: str = ATLANTIC_BASIN) -> pd.DataFrame:
    """SID / time / USA_RMW / R34 quadrants, in km, for the parametric baseline.

    Returned separately from `parse` so the corridor pipeline never carries
    columns it does not use.  Missing RMW before 1996 is expected and is what
    drives the Willoughby et al. (2006) fallback in the model stage.
    """
    df = text_or_df.copy() if isinstance(text_or_df, pd.DataFrame) \
        else read_csv(text_or_df)
    if "BASIN" in df.columns and basin:
        df = df[df["BASIN"].astype(str).str.upper() == basin.upper()]

    nm2km = 1.852
    # USA_ATCF_ID ("AL092017") is the id the hazard and damage figures key on,
    # and it is NOT the IBTrACS SID ("2017228N14314").  Carried through so the
    # analysis notebooks can join on it without re-reading the source CSV.
    atcf = (df["USA_ATCF_ID"].astype(str).str.strip().str.upper()
            if "USA_ATCF_ID" in df.columns else pd.Series("", index=df.index))
    atcf = atcf.where(~atcf.isin(["", "NAN", "NA", "MM"]), pd.NA)

    out = pd.DataFrame({
        "storm_id": df["SID"].astype(str),
        "atcf": atcf,
        "iso_time": pd.to_datetime(df["ISO_TIME"], errors="coerce"),
        "rmw_km": _num(df.get("USA_RMW")) * nm2km,
    })
    have = [c for c in R34_COLS if c in df.columns]
    if have:
        q = np.column_stack([_num(df[c]).to_numpy(float) for c in have]) * nm2km
        with np.errstate(invalid="ignore"):
            out["r34_km_mean"] = np.nanmean(q, axis=1)
            out["r34_km_max"] = np.nanmax(q, axis=1)
    else:
        out["r34_km_mean"] = np.nan
        out["r34_km_max"] = np.nan
    return out.dropna(subset=["iso_time"]).reset_index(drop=True)


def fetch(url: str = DEFAULT_URL, local: str | None = None, timeout: int = 300) -> str:
    if local and os.path.exists(local):
        return local                      # read_csv handles a path
    import requests
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def load(url: str = DEFAULT_URL, local: str | None = None,
         basin: str = ATLANTIC_BASIN) -> pd.DataFrame:
    return parse(fetch(url, local), basin=basin)


# ── old names, for the figure notebooks ──────────────────────────────────────
parse_ibtracs = parse
DEFAULT_IBTRACS_URL = DEFAULT_URL
