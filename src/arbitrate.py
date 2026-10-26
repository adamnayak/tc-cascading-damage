"""Mutually exclusive TC corridors -- sequential-storm arbitration.

THE PROBLEM
    Per-storm corridors are extracted independently, so when two TCs are close
    in space and time the same moisture can appear in BOTH: storm A leaves a
    remnant, that remnant feeds storm B, and each storm's own extraction claims
    it.  Aggregating across storms then double-counts it.

THE RULE, unchanged
    A cell claimed by more than one storm at the same frame goes to the storm
    whose CO-TEMPORAL track position is nearest, with the LATER-GENESIS storm
    winning ties.  For the sequential handoff this reproduces the intended
    behaviour automatically: once B forms, B's centre is near the moisture it is
    feeding while A has dissipated or moved on (A's centre is clamped to its last
    fix), so B wins and the parcel leaves A.  Remnant that B does not claim stays
    with A and runs its course.  For genuinely CONCURRENT storms the same rule
    splits the shared field sensibly instead of handing everything to the
    younger storm.

    Nearest-wins is a total order on distance, so applying the rule PAIRWISE is
    sufficient and correct for three-way contests: the globally nearest storm
    never loses a pairwise contest, and every other claimant loses at least one,
    so exactly one owner survives.

WHAT CHANGED
    Everything about the plumbing, nothing about the rule.

    The original `exclusive_corridor` wrote its own root, mirroring the catalog
    layout, with its own `_CLAIM_COLS`/`_EVENT_COLS`/`_coerce_links` copy of the
    link schema and a `repair_links` walker to fix catalogs written before that
    copy was pinned.  Here it produces one more lineage variant in the same
    root, and the schema lives in `tcmc.schema`.

    It also arbitrated only `lineage_flow == 1`, a magic variable name, which is
    why `catalog_model` and `catalog_aorc` both wrote an alias field called
    `lineage_flow` that was not a lineage at all.  Here the parent variant is an
    argument.

    Grid alignment is now free.  Every storm in a catalog is built on the same
    0.25 deg lattice AND with a box rule that unions the corridor, the seaboard
    clamp and the r600 disc, so two storms' grids intersect by coordinate value
    with no snapping, no `_pos_map`, and no possibility that one storm's extent
    silently clips another's.

SCOPE
    Only the parent variant's TC cells are arbitrated.  The all-interactions
    corridor is left untouched: shared parcels belong there by design.

    Ceded cells are not tagged in the parent variant; `lineage/{parent}/{sid}.nc`
    remains the reference for what a storm claimed before arbitration, and the
    ceded counts are carried in the index.
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd

from tcmc.geo import haversine_km
from tcmc.paths import Paths

TC = 1


# ══════════════════════════════════════════════════════════════════════════════
# Space-time boxes and the overlap graph
# ══════════════════════════════════════════════════════════════════════════════
def storm_boxes(paths: Paths, variant: str, storm_ids) -> pd.DataFrame:
    """Space-time bounding box per storm, read from the NC COORDS only.

    Data stays lazy: a 900-storm catalog is surveyed by reading three coordinate
    arrays per file, not three fields.
    """
    import xarray as xr
    rows = []
    for sid in storm_ids:
        fp = paths.lineage(variant, sid)
        if not os.path.exists(fp):
            continue
        with xr.open_dataset(fp) as ds:
            t = pd.to_datetime(ds["valid_time"].values)
            lo = ds["longitude"].values
            la = ds["latitude"].values
            rows.append(dict(storm_id=sid, t0=t.min(), t1=t.max(),
                             lon_min=float(lo.min()), lon_max=float(lo.max()),
                             lat_min=float(la.min()), lat_max=float(la.max())))
    return pd.DataFrame(rows)


def overlap_groups(boxes: pd.DataFrame) -> list[list[str]]:
    """Connected components of the space-time overlap graph.

    Only storms whose boxes intersect in BOTH time and space can contest a cell,
    so most storms come back as singletons needing no arbitration and the rest
    form small within-season groups.  Memory is then flat in catalog size: one
    group is held at a time.
    """
    b = boxes.reset_index(drop=True)
    n = len(b)
    if not n:
        return []
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    t0 = b["t0"].values
    t1 = b["t1"].values
    lo0, lo1 = b["lon_min"].values, b["lon_max"].values
    la0, la1 = b["lat_min"].values, b["lat_max"].values
    order = np.argsort(t0)
    for x in range(n):
        i = order[x]
        for y in range(x + 1, n):
            j = order[y]
            if t0[j] > t1[i]:
                break                    # sorted by t0: no later storm can overlap i
            if (lo0[j] <= lo1[i] and lo0[i] <= lo1[j]
                    and la0[j] <= la1[i] and la0[i] <= la1[j]):
                union(i, j)

    comp: dict[int, list[str]] = {}
    for i in range(n):
        comp.setdefault(find(i), []).append(str(b.loc[i, "storm_id"]))
    return [sorted(v) for v in comp.values()]


# ══════════════════════════════════════════════════════════════════════════════
# Arbitration
# ══════════════════════════════════════════════════════════════════════════════
def _pos_map(arr, vals) -> np.ndarray:
    """Index of each `vals` entry within `arr`, matched on exact value.

    Exact because every storm's grid is built by `tcmc.fields.grid` from the same
    integer lattice, so two storms' shared cells have identical float64 centres.
    The original had to round to four decimals because three different catalogs
    computed their coordinates independently.
    """
    pos = {v: i for i, v in enumerate(np.asarray(arr).tolist())}
    return np.array([pos[v] for v in np.asarray(vals).tolist()], dtype=int)


def _load(paths: Paths, variant: str, sid: str) -> dict:
    import xarray as xr
    with xr.open_dataset(paths.lineage(variant, sid)) as ds:
        return dict(tc=(ds["lineage"].values == TC),
                    lons=ds["longitude"].values,
                    lats=ds["latitude"].values,
                    times=pd.to_datetime(ds["valid_time"].values),
                    lineage=ds["lineage"].values.copy())


def resolve_group(sids, tracks: dict, paths: Paths, variant: str, *,
                  tie_later_genesis: bool = True) -> dict[str, np.ndarray]:
    """Mutually exclusive TC masks for one overlap group, on each storm's own grid."""
    from tcmc.corridor.marker import track_centers

    data = {sid: _load(paths, variant, sid) for sid in sids}
    excl = {sid: data[sid]["tc"].copy() for sid in sids}
    if len(sids) == 1:
        return excl

    genesis = {sid: pd.to_datetime(tracks[sid]["iso_time"]).min() for sid in sids}
    centres = {}
    for sid in sids:
        d = data[sid]
        c = track_centers(tracks[sid], d["lons"], d["lats"], d["times"])
        centres[sid] = (np.asarray(c["lat"], float), np.asarray(c["lon"], float))

    for a, bb in itertools.combinations(sids, 2):
        A, B = data[a], data[bb]
        ct = np.intersect1d(A["times"].values, B["times"].values)
        if ct.size == 0:
            continue
        clon = np.intersect1d(A["lons"], B["lons"])
        clat = np.intersect1d(A["lats"], B["lats"])
        if clon.size == 0 or clat.size == 0:
            continue

        At = _pos_map(A["times"].values, ct)
        Bt = _pos_map(B["times"].values, ct)
        Ai, Aj = _pos_map(A["lats"], clat), _pos_map(A["lons"], clon)
        Bi, Bj = _pos_map(B["lats"], clat), _pos_map(B["lons"], clon)

        contested = A["tc"][np.ix_(At, Ai, Aj)] & B["tc"][np.ix_(Bt, Bi, Bj)]
        if not contested.any():
            continue

        LON2, LAT2 = np.meshgrid(clon, clat)
        a_later = genesis[a] > genesis[bb]
        for k in range(len(ct)):
            cm = contested[k]
            if not cm.any():
                continue
            ii, jj = np.nonzero(cm)
            cy, cx = LAT2[cm], LON2[cm]
            da = haversine_km(cy, cx, centres[a][0][At[k]], centres[a][1][At[k]])
            db = haversine_km(cy, cx, centres[bb][0][Bt[k]], centres[bb][1][Bt[k]])
            a_wins = da < db
            tie = da == db
            if tie.any() and tie_later_genesis:
                a_wins = a_wins | (tie if a_later else False)
            la = ~a_wins
            if la.any():
                excl[a][At[k], Ai[ii[la]], Aj[jj[la]]] = False
            if a_wins.any():
                excl[bb][Bt[k], Bi[ii[a_wins]], Bj[jj[a_wins]]] = False
    return excl


def apply_mask(paths: Paths, parent_variant: str, sid: str,
               mask: np.ndarray) -> np.ndarray:
    """Parent lineage with ceded TC cells demoted to 0.

    Foreign lineages are left exactly as they were: arbitration decides which TC
    owns a shared parcel, never whether a front owns one.
    """
    d = _load(paths, parent_variant, sid)
    lin = d["lineage"]
    out = np.where((lin == TC) & ~mask, 0, lin).astype(np.int16)
    return out


def group_report(sids, before: dict, after: dict) -> pd.DataFrame:
    """Per-storm cell counts before and after, so the cession is auditable."""
    rows = []
    for sid in sids:
        b = int(before[sid].sum())
        a = int(after[sid].sum())
        rows.append(dict(storm_id=sid, group_size=len(sids),
                         contested=len(sids) > 1, tc_cells=b,
                         exclusive_cells=a, ceded_cells=b - a,
                         ceded_frac=round(1 - a / b, 4) if b else 0.0))
    return pd.DataFrame(rows)
