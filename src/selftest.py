"""Corridor self-test on a synthetic storm.  No network, no ERA5.

    python -m tcmc.corridor.selftest

Builds a two-plume IVT field where a TC-seeded plume and an independent front
approach, merge, and separate, then checks the properties the science depends
on.  The synthetic field exists so these can be asserted rather than eyeballed:

  * detection is a function of |IVT| alone, so stage 2 is offline-reproducible
  * `h -> inf` reproduces connected components EXACTLY, which is what makes the
    watershed a sensitivity axis rather than a new method
  * basins exactly refine components -- no basin crosses a component boundary,
    no masked cell is unassigned -- which is the containment guarantee the
    subset property and the branch aggregates both rest on
  * basin labels PERSIST across frames, so the basin id is the branch id
  * `stem` and `stem_ws` are both subsets of `all`
  * with no subdivision, `stem_ws` is bit-identical to `stem`
  * the undirected ablation admits the front's pre-merger history and the
    directed graph does not -- the delta the construction exists to produce
  * generation separates the TC core from a parcel fed through a front
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from tcmc.corridor import dominance as D
from tcmc.corridor import events as EV
from tcmc.corridor import lineage as LIN
from tcmc.corridor import marker as MK
from tcmc.corridor import segment as SEG
from tcmc.corridor import thr as THR
from tcmc.geo import cell_area_km2

_FAILED: list[str] = []


def check(cond: bool, what: str) -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {what}")
    if not cond:
        _FAILED.append(what)


# ══════════════════════════════════════════════════════════════════════════════
def synthetic(T: int = 20, ny: int = 110, nx: int = 150):
    """Two plumes that approach, merge, and separate again.

    The TC plume tracks northeast out of the Gulf and carries the marker.  The
    front descends from 47 N to 35 N over the first twelve frames, joins the TC
    plume there, then recedes to 46 N -- a V in latitude, so the sequence
    contains a genuine merge AND a genuine separation.  That matters: the merge
    exercises the dominance test and tau, and the separation spawns a branch
    from a TC-owned parent, which is the only way a non-zero GENERATION can
    arise.

    Amplitudes are set so the front out-masses the TC while they are joined, so
    the content-dominance test has something to decide rather than rubber-
    stamping the marker.  The marker radius is kept to 150 km so the disc stays
    inside the TC basin and does not seed the front's basin by accident, which
    would collapse every generation to zero.
    """
    lat = np.linspace(51.0, 51.0 - 0.25 * (ny - 1), ny)      # descending, as ERA5
    lon = np.linspace(-102.0, -102.0 + 0.25 * (nx - 1), nx)
    Lon, Lat = np.meshgrid(lon, lat)
    times = pd.date_range("2012-09-01", periods=T, freq="6h")

    tc_lat = np.linspace(25.0, 38.5, T)
    tc_lon = np.linspace(-93.0, -76.0, T)
    fr_lat = np.concatenate([np.linspace(47.0, 35.0, 12),
                             np.linspace(35.0, 46.0, T - 12)])
    fr_lon = np.linspace(-99.0, -70.0, T)

    f = np.zeros((T, ny, nx), np.float32)
    for k in range(T):
        f[k] += 250.0 * np.exp(-(((Lat - tc_lat[k]) / 1.9) ** 2
                                 + ((Lon - tc_lon[k]) / 2.2) ** 2))
        f[k] += 310.0 * np.exp(-(((Lat - fr_lat[k]) / 1.7) ** 2
                                 + ((Lon - fr_lon[k]) / 3.0) ** 2))
        f[k] += 55.0                                       # broad background
    track = pd.DataFrame({"iso_time": times, "lat": tc_lat, "lon": tc_lon})
    iwv = (f / 8.0).astype(np.float32)
    return dict(ivt=f, lat=lat, lon=lon, times=times, track=track,
                iwv=iwv, marker_radius_km=150.0)


def detect(s, **kw):
    p = dict(s=12, t=0, floor=30.0, min_area_km2=1.0e4, open_radius=1)
    p.update(kw)
    return THR.detect_all(s["ivt"], s["lat"], s["lon"], **p)


# ══════════════════════════════════════════════════════════════════════════════
def test_detection(s):
    print("\ndetection")
    d = detect(s)
    check(d["anomaly"].min() >= 0, "the THR anomaly is non-negative")
    check(d["mask"].any(), "the mask is non-empty on the synthetic field")
    check(d["component"].max() >= 2, "two systems are detected as separate components")

    d2 = THR.detect_all(s["ivt"] + 500.0, s["lat"], s["lon"], s=12, t=0,
                        floor=30.0, min_area_km2=1.0e4, open_radius=1)
    check(np.array_equal(d["mask"], d2["mask"]),
          "detection is invariant to a uniform background offset "
          "(the point of working on an anomaly)")

    A = cell_area_km2(s["lat"], s["lon"])
    big = THR.detect(d["anomaly"], A, floor=30.0, min_area_km2=1e9, open_radius=1)
    check(not big.any(), "min_area_km2 deletes everything when set absurdly high "
                         "(the IPART 5e5 failure mode, made visible)")
    return d


def test_segmentation(s, d):
    print("\nsegmentation")
    A = d["area_km2"]
    lo = SEG.basins(d["anomaly"], d["mask"], d["component"], A,
                    enabled=True, h=8.0, min_basin_km2=5e3)
    hi = SEG.basins(d["anomaly"], d["mask"], d["component"], A,
                    enabled=True, h=1e9, min_basin_km2=0.0)
    off = SEG.basins(d["anomaly"], d["mask"], d["component"], A, enabled=False,
                     min_basin_km2=0.0)

    def same_partition(a, b):
        """Identical partitions up to relabelling.  Checked BOTH ways.

        A one-way check only proves refinement, which every watershed output
        satisfies trivially, and would have passed while the partitions differed.
        """
        for k in range(a.shape[0]):
            if not np.array_equal(a[k] > 0, b[k] > 0):
                return False
            for src, dst in ((a[k], b[k]), (b[k], a[k])):
                for v in np.unique(src):
                    if v == 0:
                        continue
                    if len(np.unique(dst[src == v])) != 1:
                        return False
        return True

    check(same_partition(hi["basin"], d["component"]),
          "h -> inf reproduces connected components EXACTLY")
    check(same_partition(off["basin"], d["component"]),
          "watershed disabled reproduces connected components EXACTLY")
    check(lo["stats"]["basins_per_component"] > 1.10,
          f"a small h subdivides merged components: "
          f"{lo['stats']['basins_per_component']:.2f} basins per component")
    ns = int(SEG.subdivided_components(d["component"], lo["basin"]).sum())
    check(ns > 0, f"{ns} component-frames were actually split "
                  "(a run where this is 0 is identical to the non-watershed pipeline, "
                  "whatever h was set to)")

    # containment: basins refine components, nothing crosses, nothing is lost
    ok_refine, ok_cover = True, True
    for k in range(d["mask"].shape[0]):
        for b in np.unique(lo["basin"][k]):
            if b == 0:
                continue
            if len(np.unique(d["component"][k][lo["basin"][k] == b])) != 1:
                ok_refine = False
        if not np.array_equal(lo["basin"][k] > 0, d["mask"][k]):
            ok_cover = False
    check(ok_refine, "no basin crosses a component boundary")
    check(ok_cover, "every masked cell belongs to exactly one basin")

    # persistence: the same physical parcel keeps its id
    shared = []
    for k in range(1, d["mask"].shape[0]):
        a = set(np.unique(lo["basin"][k - 1])) - {0}
        b = set(np.unique(lo["basin"][k])) - {0}
        if a and b:
            shared.append(len(a & b) / len(a | b))
    check(np.mean(shared) > 0.70,
          f"basin labels persist across frames (mean Jaccard {np.mean(shared):.2f}) "
          "-- the basin id IS the branch id")

    live = [len(np.unique(lo["basin"][k])) - 1 for k in range(d["mask"].shape[0])]
    check(max(live) < 40, f"h controls over-segmentation (max {max(live)} basins/frame)")
    return lo, hi


def test_lineage(s, d, seg_lo, seg_hi):
    print("\nlineage")
    M = MK.marker_mask(s["track"], s["lon"], s["lat"], s["times"], radius_km=s["marker_radius_km"])
    check(M.any(), "the marker disc is non-empty")
    A = d["area_km2"]
    kw = dict(measure="content", frac=0.5, iwv=s["iwv"], area=A)

    r_all = LIN.build_variant("all", {"component": d["component"],
                                      "basin": seg_lo["basin"]}, M, **kw)
    r_und = LIN.build_variant("undirected", {"component": d["component"],
                                             "basin": seg_lo["basin"]}, M, **kw)
    r_stem = LIN.build_variant("stem", {"component": d["component"],
                                        "basin": seg_lo["basin"]}, M, **kw)
    r_ws = LIN.build_variant("stem_ws", {"component": d["component"],
                                         "basin": seg_lo["basin"]}, M,
                             only_at_merge=False, **kw)
    r_ws_hi = LIN.build_variant("stem_ws", {"component": d["component"],
                                            "basin": seg_hi["basin"]}, M,
                                only_at_merge=False, **kw)

    a = LIN.corridor_of(r_all["lineage"])
    u = LIN.corridor_of(r_und["lineage"])
    st = LIN.corridor_of(r_stem["lineage"])
    ws = LIN.corridor_of(r_ws["lineage"])

    check(a.any() and st.any(), "both graphs produce a non-empty corridor")
    check(int((st & ~a).sum()) == 0, "`stem` is a SUBSET of `all`")
    check(int((ws & ~a).sum()) == 0, "`stem_ws` is a SUBSET of `all`")
    check(LIN.pruned_fraction(r_all["lineage"], r_ws["lineage"], A)
          ["subset_violations"] == 0, "the subset property is asserted, not assumed")

    check(np.array_equal(LIN.corridor_of(r_ws_hi["lineage"]), st),
          "with no subdivision, `stem_ws` is bit-identical to `stem`")

    check(int(u.sum()) >= int(a.sum()),
          "the undirected ablation claims at least as much as the directed graph")
    delta = int(u.sum()) - int(a.sum())
    check(delta > 0,
          f"undirected admits the front's pre-merger history: "
          f"{delta:,} extra cell-frames awarded to the TC")

    # the front must exist as its own lineage before it joins
    pre = (r_all["lineage"] > LIN.TC)
    check(pre.any(), "a foreign lineage exists before it merges")
    first_tc = int(np.argmax(a.reshape(a.shape[0], -1).any(1)))
    first_other = int(np.argmax(pre.reshape(pre.shape[0], -1).any(1)))
    check(first_other <= first_tc + 2,
          "the front is tracked from its own birth, not from the merge")

    mg = EV.merge_table(r_all["events"], r_all["lineage"], s["times"], s["lat"],
                        s["lon"], A, basin=seg_lo["basin"])
    check(len(mg) >= 1, f"a merge into the TC is recorded (tau={list(mg['tau'])[:1]})")
    if len(mg):
        check(bool(mg["frames_before_tau"].iloc[0] > 0),
              "the merged lineage has a pre-tau lifetime, which is what an "
              "undirected corridor would have given to the TC")

    pts = pd.DataFrame({
        "time": [s["times"][2], s["times"][2], s["times"][-1]],
        "lineage": [2, 2, 2]})
    pt = EV.pre_tau_counts(pts, mg)
    check(len(pt) and int(pt["n_pre_tau"].sum()) >= 1,
          "pre_tau_counts returns the misattribution count")

    ar = LIN.area_stats(r_stem["lineage"], A)
    check(0 < ar["area_frac_box"] < 1, "corridor area fraction is in range")
    return r_all, r_stem, r_ws, seg_lo, M


def test_branches(s, d, r_ws, seg_lo, M):
    print("\nbranches and generation")
    A = d["area_km2"]
    bt = EV.branch_table(seg_lo["basin"], r_ws["lineage"], M, A, s["lat"], s["lon"],
                         ivt=s["ivt"], iwv=s["iwv"], variant="stem_ws")
    check(len(bt) > 0, f"{len(bt)} branches tabulated")
    check(set(bt["owner"]) <= {"tc", "foreign", "none"}, "owner is a clean category")
    check(bool((bt["owner"] == "tc").any()), "some branches are TC-owned")
    check(bool((bt["death_frame"] >= bt["birth_frame"]).all()),
          "every branch dies no earlier than it is born")
    check(bool((bt["n_frames"] > 0).all()), "every branch lives at least one frame")

    g = EV.generations(seg_lo["basin"], M)
    vals = sorted({v for v in g.values()})
    check(0 in vals, "some branch is marker-seeded (generation 0)")
    check(max(vals) >= 1,
          f"generation separates downstream parcels from the core: {vals[:6]} "
          "(0 = marker-seeded, 1 = fed directly by it)")
    check(all(v >= -1 for v in vals), "unreachable branches are -1, not 0")

    tc = bt[bt["owner"] == "tc"]
    check(bool((tc["generation"] >= 0).all()),
          "no TC-owned branch is marked unreachable")

    stem = bt[(bt["owner"] == "tc") & (bt["generation"] == 0)]
    check(len(stem) >= 1,
          f"the MAIN STEM is well defined: {len(stem)} generation-0 TC branch(es)")
    check(float(bt["area_km2_mean"].sum()) > 0, "branch areas are populated")
    check(bool(np.isfinite(bt["iwv_mass_kg_mean"]).all()),
          "branch IWV mass is populated when the field is supplied")


def test_dominance(s, d):
    print("\ndominance")
    A = d["area_km2"]
    lab = d["component"][6]
    m = D.component_measure(lab, s["iwv"][6], A, "content")
    check(len(m) >= 1 and all(v > 0 for v in m.values()),
          "content measure returns a positive mass per object")
    pk = D.component_measure(lab, s["iwv"][6], A, "peak")
    check(all(pk[k] <= m[k] for k in pk if A.sum() > 1),
          "peak intensity is not a mass (the two measures are distinct)")

    w, share, den = D.resolve_owner({1: 9.0, 2: 1.0}, 0.5)
    check(w == 1 and abs(share - 0.9) < 1e-9, "a clear majority wins")
    w, _, _ = D.resolve_owner({1: 5.0, 2: 5.0}, 0.6)
    check(w is None, "a genuine tie under the threshold is CONTESTED, not awarded")
    w, _, _ = D.resolve_owner({3: 1.0}, 0.9)
    check(w == 3, "a sole parent CONTINUES regardless of the threshold "
                  "(a coasting remnant keeps its identity)")
    w, share, _ = D.resolve_owner({1: 6.0, 2: 4.0}, 0.5, ambient=10.0,
                                  include_ambient=True)
    check(w is None and abs(share - 0.3) < 1e-9,
          "ambient inflow dilutes the share in flux mode but can never own a parcel")

    # Parent and child must ADJOIN: `edge_fluxes` credits the label occupying the
    # cell just outside the child's boundary, so a gap between them would credit
    # ambient.  A real parent (frame k-1) and child (frame k) overlap, which is
    # the case reproduced here.
    U = np.zeros((20, 20), np.float32)
    V = np.zeros((20, 20), np.float32)
    U[:, :] = 50.0                                     # flow to the east
    Lp = np.zeros((20, 20), np.int32); Lp[5:15, 2:10] = 1
    Lc = np.zeros((20, 20), np.int32); Lc[5:15, 9:16] = 2
    fx = D.edge_fluxes(Lp, Lc, U, V)
    check(any(k[0] == 1 for k in fx),
          "directed flux credits an upwind parent that is feeding the child")
    fx2 = D.edge_fluxes(Lp, Lc, np.zeros_like(U), np.full_like(V, 50.0))
    into = sum(v for k, v in fx.items() if k[0] == 1)
    along = sum(v for k, v in fx2.items() if k[0] == 1)
    check(along < into,
          "flow ALONG the interface contributes less than flow ACROSS it "
          "(the reason overlap area is not the edge weight)")


def test_courant(s, d):
    print("\ndiagnostics")
    c = MK.track_centers(s["track"], s["lon"], s["lat"], s["times"])
    check(bool(c["in_track"].all()), "every synthetic frame is in-track")
    cl = MK.courant(d["mask"], c["i"], c["j"])
    fin = cl[np.isfinite(cl)]
    check(fin.size and float(np.nanmax(fin)) < 1.0,
          f"C_link stays below 1 at 6-hourly (max {np.nanmax(fin):.2f}): "
          "overlap linking is safe here")
    es = MK.edge_stats(d["mask"], s["lat"], s["lon"])
    check("edge_frames" in es and es["edge_frames"] >= 0,
          f"edge diagnostics computed ({es['edge_frames']} frames touch an edge)")


# ══════════════════════════════════════════════════════════════════════════════
def main() -> int:
    print("tcmc corridor self-test  (synthetic two-plume merging storm)")
    s = synthetic()
    d = test_detection(s)
    seg_lo, seg_hi = test_segmentation(s, d)
    r_all, r_stem, r_ws, seg_lo, M = test_lineage(s, d, seg_lo, seg_hi)
    test_branches(s, d, r_ws, seg_lo, M)
    test_dominance(s, d)
    test_courant(s, d)

    print()
    if _FAILED:
        print(f"{len(_FAILED)} CHECK(S) FAILED:")
        for f in _FAILED:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
