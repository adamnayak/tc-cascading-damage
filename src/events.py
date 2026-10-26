"""What was measured: tau, junctions, pre-tau misattribution, and branches.

TWO TABLES

`merge_table`  one row per foreign LINEAGE that joins the TC lineage:
        tau_j              when
        junction lat/lon   where, and whether it is inside the plotted frame
        area / lifetime before tau_j
        hazards before tau_j   <- what an undirected corridor awards to the TC

    That last quantity is the number the directed construction exists to
    produce.  `pre_tau_counts` computes it against any hazard point table.

`branch_table`  one row per tracked watershed basin, per variant:
        branch_id, lineage, owner, generation, parent_branch, merged_into,
        birth/death/tau frames, area and IWV statistics, centroid

    This is the new table.  A branch is a persistent basin identity produced by
    `tcmc.corridor.segment`; this module attaches ownership, lifetime and
    generation to it so the precipitation aggregates have something to group on.

TAU UNDER WATERSHEDDING -- a definition that had to change
    With component-level ownership a foreign lineage joins the TC at a single
    frame: the whole blob changes hands and tau is unambiguous.  With basin-level
    ownership a front never wholly joins -- some of its basins do and some do
    not, possibly at different frames.

    tau_j is therefore defined as the FIRST frame at which ANY object of lineage
    j is TC-owned, and `frames_before_tau` and `area_before_tau_km2` are
    measured over lineage j as a whole.  Under component granularity this is
    identical to the original definition, so the `stem` and `all` numbers are
    unchanged; under `stem_ws` it is the weakest defensible reading, and
    `n_basins_joined` alongside it says how much of the front actually changed
    hands.  This must be restated in the methods; it is not a free refinement.

GENERATION
    Distance from the TC core in the branch graph: 0 for any branch that is ever
    marker-seeded, and 1 + the minimum generation of its detected parents
    otherwise.  Computed by a forward breadth-first pass over the branch graph,
    so a branch can never inherit a generation from its own future -- the same
    causal restriction that makes the lineage graph directed.  A branch that
    never connects to the TC keeps generation -1, which is the honest value:
    "not fed by the TC", distinct from "fed at distance 0".
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from tcmc.corridor.segment import overlap_pairs

TC = 1

_MERGE_COLS = ["lineage", "tau", "frame", "junction_lat", "junction_lon",
               "junction_in_frame", "frames_before_tau", "area_before_tau_km2",
               "n_basins_joined"]


# ══════════════════════════════════════════════════════════════════════════════
def merge_table(events: pd.DataFrame, lineage: np.ndarray, times,
                lat: np.ndarray, lon: np.ndarray, area_km2: np.ndarray,
                bbox: dict | None = None, basin: np.ndarray | None = None
                ) -> pd.DataFrame:
    """One row per foreign lineage joining the TC.

    `junction_in_frame` answers the video complaint directly: a corridor that
    joins outside the plotted bbox is LABELLED, not silently drawn.
    """
    if events is None or not len(events) or "event" not in events:
        return pd.DataFrame(columns=_MERGE_COLS)
    m = events[(events["event"] == "merge") & (events.get("into", -1) == TC)]
    if not len(m):
        return pd.DataFrame(columns=_MERGE_COLS)

    times = pd.to_datetime(np.asarray(times))
    rows = []
    for j, g in m.groupby("lineage"):
        k = int(g["frame"].min())
        pre = lineage[:k] == j
        prev = lineage[k - 1] == j if k else np.zeros(lineage.shape[1:], bool)
        if prev.any():
            ci, cj = ndi.center_of_mass(prev)
        else:
            ci, cj = np.array(lineage.shape[1:]) / 2.0
        jlat = float(lat[int(round(ci))])
        jlon = float(lon[int(round(cj))])
        inf = True if bbox is None else (
            bbox["lon_min"] <= jlon <= bbox["lon_max"]
            and bbox["lat_min"] <= jlat <= bbox["lat_max"])

        n_joined = 0
        if basin is not None and k:
            was = set(np.unique(basin[k - 1][prev])) - {0}
            now = set(np.unique(basin[k][lineage[k] == TC])) - {0}
            n_joined = len(was & now)

        rows.append(dict(
            lineage=int(j), tau=times[k], frame=k,
            junction_lat=jlat, junction_lon=jlon, junction_in_frame=bool(inf),
            frames_before_tau=int(pre.reshape(k, -1).any(1).sum()) if k else 0,
            area_before_tau_km2=float(
                np.mean([(area_km2 * pre[q]).sum() for q in range(k)]) if k else 0.0),
            n_basins_joined=int(n_joined)))
    return pd.DataFrame(rows, columns=_MERGE_COLS).sort_values("tau") \
                                                  .reset_index(drop=True)


def pre_tau_counts(points: pd.DataFrame, merges: pd.DataFrame, *,
                   time_col: str = "time",
                   lineage_col: str = "lineage") -> pd.DataFrame:
    """THE NUMBER.  Hazards in a foreign lineage's object BEFORE it joined the TC.

    An undirected corridor awards every one of these to the tropical cyclone.
    Summed over a catalog, this is the misattribution the directed construction
    removes, as a count rather than as an argument.
    """
    cols = ["lineage", "tau", "n_pre_tau", "n_post_tau"]
    if points is None or not len(points) or merges is None or not len(merges):
        return pd.DataFrame(columns=cols)
    t = pd.to_datetime(points[time_col])
    rows = []
    for m in merges.itertuples():
        sub = points[points[lineage_col] == m.lineage]
        if not len(sub):
            continue
        ts = pd.to_datetime(sub[time_col])
        rows.append(dict(lineage=int(m.lineage), tau=m.tau,
                         n_pre_tau=int((ts < m.tau).sum()),
                         n_post_tau=int((ts >= m.tau).sum())))
    return pd.DataFrame(rows, columns=cols)


# ══════════════════════════════════════════════════════════════════════════════
def branch_graph(basin: np.ndarray, *, min_overlap: int = 1, dilate: int = 1
                 ) -> dict[int, set[int]]:
    """{child branch: set of parent branches}, forward in time.

    A branch is usually its own parent (labels persist), which is what makes a
    continuation edge trivial; the interesting edges are the ones where a
    different id feeds in, i.e. a merge or a split.
    """
    edges: dict[int, set[int]] = defaultdict(set)
    for k in range(1, basin.shape[0]):
        for (p, c) in overlap_pairs(basin[k - 1], basin[k],
                                    min_overlap=min_overlap, dilate=dilate):
            if p and c and p != c:
                edges[int(c)].add(int(p))
    return dict(edges)


def generations(basin: np.ndarray, marker: np.ndarray, *, min_overlap: int = 1,
                dilate: int = 1) -> dict[int, int]:
    """Branch -> graph distance from a marker-seeded branch.  -1 if unreachable.

    Forward breadth-first over the branch graph.  A branch that has ever
    contained a marker cell is generation 0; one fed directly by such a branch
    is 1; one fed through an intermediate parcel is 2.  This is the quantity
    that answers "downstream lineage from moisture parcel interactions versus
    the primary TC corridor" directly, because it distinguishes moisture the TC
    delivered from moisture a front delivered into a parcel the TC also touched.
    """
    seeded: set[int] = set()
    for k in range(basin.shape[0]):
        if marker[k].any():
            seeded |= {int(v) for v in np.unique(basin[k][marker[k]]) if v > 0}

    parents = branch_graph(basin, min_overlap=min_overlap, dilate=dilate)
    children: dict[int, set[int]] = defaultdict(set)
    for c, ps in parents.items():
        for p in ps:
            children[p].add(c)

    all_ids = {int(v) for v in np.unique(basin) if v > 0}
    gen = {b: -1 for b in all_ids}
    q: deque[int] = deque()
    for b in seeded:
        gen[b] = 0
        q.append(b)
    while q:
        b = q.popleft()
        for c in children.get(b, ()):
            if gen.get(c, -1) == -1:
                gen[c] = gen[b] + 1
                q.append(c)
    return gen


# ══════════════════════════════════════════════════════════════════════════════
def branch_table(basin: np.ndarray, lineage: np.ndarray, marker: np.ndarray,
                 area_km2: np.ndarray, lat: np.ndarray, lon: np.ndarray, *,
                 ivt: np.ndarray | None = None, iwv: np.ndarray | None = None,
                 variant: str = "", min_overlap: int = 1, dilate: int = 1
                 ) -> pd.DataFrame:
    """One row per tracked basin, with its ownership under `lineage`.

    Geometry (birth, death, area, centroid, parents) comes from `basin` and is
    variant-independent; ownership, generation and tau come from `lineage` and
    are not, which is why the table is written per variant.
    """
    T = basin.shape[0]
    ids = [int(v) for v in np.unique(basin) if v > 0]
    if not ids:
        return pd.DataFrame(columns=_branch_cols())

    gen = generations(basin, marker, min_overlap=min_overlap, dilate=dilate)
    parents = branch_graph(basin, min_overlap=min_overlap, dilate=dilate)

    # `merged_into`: after a branch's last frame, which branch occupies its
    # cells?  Recorded so a disappearing parcel can be followed rather than
    # simply vanishing from the lifetime statistics.
    merged_into: dict[int, int] = {}
    for b in ids:
        frames = np.flatnonzero((basin == b).reshape(T, -1).any(1))
        if not frames.size or frames[-1] + 1 >= T:
            continue
        kl = int(frames[-1])
        nxt = basin[kl + 1][basin[kl] == b]
        nxt = nxt[nxt > 0]
        if nxt.size:
            vals, cnt = np.unique(nxt, return_counts=True)
            merged_into[b] = int(vals[int(np.argmax(cnt))])

    Lon, Lat = np.meshgrid(np.asarray(lon, float), np.asarray(lat, float))
    rows = []
    for b in ids:
        m = basin == b
        live = np.flatnonzero(m.reshape(T, -1).any(1))
        k0, k1 = int(live[0]), int(live[-1])

        areas = np.array([float(area_km2[m[k]].sum()) for k in live])
        lin_vals = lineage[m]
        lin_vals = lin_vals[lin_vals > 0]
        own = int(np.bincount(lin_vals).argmax()) if lin_vals.size else 0

        tc_frames = np.array([bool((lineage[k][m[k]] == TC).any()) for k in live])
        tau = int(live[int(np.argmax(tc_frames))]) if tc_frames.any() else -1

        any_cells = m.any(axis=0)
        rows.append(dict(
            variant=variant, branch_id=int(b), lineage=int(own),
            owner=("tc" if own == TC else ("foreign" if own else "none")),
            generation=int(gen.get(b, -1)),
            parent_branch=int(min(parents.get(b, {0}))) if parents.get(b) else 0,
            birth_frame=k0, death_frame=k1, n_frames=int(live.size),
            tau_frame=tau, merged_into=int(merged_into.get(b, 0)),
            area_km2_mean=float(areas.mean()), area_km2_max=float(areas.max()),
            iwv_mass_kg_mean=float(np.mean(
                [float((iwv[k] * area_km2 * 1e6)[m[k]].sum()) for k in live]))
            if iwv is not None else np.nan,
            ivt_mean=float(np.mean([float(ivt[k][m[k]].mean()) for k in live]))
            if ivt is not None else np.nan,
            centroid_lat=float(Lat[any_cells].mean()),
            centroid_lon=float(Lon[any_cells].mean()),
        ))
    return pd.DataFrame(rows, columns=_branch_cols())


def _branch_cols() -> list[str]:
    from tcmc.schema import BRANCHES
    return [c for c in BRANCHES if c not in ("storm_id", "name")]


# ══════════════════════════════════════════════════════════════════════════════
def summarize(name: str, variant: str, res: dict, merges: pd.DataFrame,
              branches: pd.DataFrame | None = None) -> str:
    """One readable block per storm.  Used by the progress reporter and by hand."""
    lines = [f"{name}  [{variant}]"]
    a = res.get("area_frac_box")
    if a is not None:
        lines.append(f"  corridor {100 * a:5.2f}% of box | "
                     f"{res.get('n_lineages', 0)} lineages | "
                     f"{len(merges)} merged into the TC")
    if branches is not None and len(branches):
        tc = branches[branches["owner"] == "tc"]
        g = tc["generation"]
        lines.append(f"  branches {len(branches)} ({len(tc)} TC-owned) | "
                     f"generation "
                     f"{'/'.join(str(int(x)) for x in sorted(g.unique())[:5])}")
    if len(merges):
        lines.append("  MERGES (a foreign lineage joins the TC)")
        lines.append("    " + merges.to_string(index=False).replace("\n", "\n    "))
    else:
        lines.append("  no foreign lineage merged into this TC -- an isolated corridor")
    return "\n".join(lines)
