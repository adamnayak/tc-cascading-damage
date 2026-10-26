"""Stage 2b -- ownership.  Which lineage owns which object, forward in time.

THE CONTRIBUTION -- the causal ordering
    Edges are FORWARD-DIRECTED.  Undirected connectivity is an equivalence
    relation -- reflexive, symmetric, transitive -- and symmetry is the bug: a
    breadth-first search walks BACKWARD through a merged component and admits a
    frontal system's entire pre-merger history into the TC corridor, together
    with every claim and tornado it caused.  That is what 3-D space-time
    connected components do.  Defensible for climatology, where a merged object
    IS the object of interest.  Not defensible for per-event attribution.

    `mode="undirected"` restores the old behaviour so the fix is a measured
    delta rather than an assertion.

FOUR VARIANTS, ONE SWEEP
    The sweep is parameterised by what it labels and how it decides.

        variant     mode           granularity   decision at a merge
        --------------------------------------------------------------------
        all         reachability   component     compound: the TC absorbs it
        undirected  undirected     component     (ablation) everything, always
        stem        dominance      component     the dominant feeder takes the
                                                 whole component
        stem_ws     dominance      basin         the dominant feeder takes each
                                                 BASIN it dominantly feeds

    `stem_ws` is the new one.  Under `stem` the decision is all-or-nothing for
    a merged blob; under `stem_ws` the blob has already been subdivided by
    `tcmc.corridor.segment`, so the TC keeps the basins it dominantly feeds and
    cedes the rest.  Because basins exactly refine components, `stem_ws` is
    still a subset of `all`, so the pruned set stays well defined.

    The two granularities run the SAME function over a different label field.
    With `watershed.h` large enough that no component is subdivided, basin and
    component labelling coincide and `stem_ws` reproduces `stem` exactly, which
    is the property that makes `h` a sensitivity axis rather than a new method.

    `only_at_merge` is applied here, not in segmentation: a component with
    fewer than two inbound lineages is decided as a unit even under basin
    granularity, so a lone parcel is never split just because its anomaly field
    is bumpy.

CONVENTION
    After tau_j the coupled system is the TC's (the COMPOUND convention) under
    `all`.  Reported as a convention, never assumed as a fact.  Under `stem` and
    `stem_ws` the convention is replaced by the dominance test, which is the
    whole point of having both.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from tcmc.corridor import dominance as D
from tcmc.corridor.segment import overlap_pairs

TC = 1

#: variant -> (mode, granularity)
VARIANT_SPEC: dict[str, tuple[str, str]] = {
    "all": ("reachability", "component"),
    "undirected": ("undirected", "component"),
    "stem": ("dominance", "component"),
    "stem_ws": ("dominance", "basin"),
}


# ══════════════════════════════════════════════════════════════════════════════
def build(labels: np.ndarray, marker: np.ndarray, *, mode: str = "reachability",
          measure: str = "content", frac: float = D.MAIN_FEEDER_FRAC,
          iwv: np.ndarray | None = None, uv: tuple | None = None,
          area: np.ndarray | None = None, component: np.ndarray | None = None,
          only_at_merge: bool = False, min_overlap: int = 1,
          dilate: int = 1) -> dict:
    """Forward-directed ownership over `labels`.  One sweep, four modes.

    Parameters
    ----------
    labels : (T, Y, X) int, the objects being owned.  Components for the
        `all`/`stem` variants, basins for `stem_ws`.
    marker : (T, Y, X) bool, the track disc.  An object containing a marker cell
        is the TC's, unconditionally: the track is ground truth.
    component : needed only when `only_at_merge` is set under basin
        granularity, to decide whether a component is at a genuine merge.

    Returns
    -------
    lineage (int16), events (DataFrame), n_lineages, ambient_share
    """
    if mode == "undirected":
        return _undirected(labels, marker, min_overlap, dilate)

    T = labels.shape[0]
    own: list[dict[int, int]] = [dict() for _ in range(T)]
    events: list[dict] = []
    next_lin = TC + 1
    amb_num = amb_den = 0.0
    use_flux = (mode == "dominance" and measure == "flux")
    if mode == "dominance" and measure != "flux" and iwv is None:
        raise ValueError(f"measure={measure!r} needs the IWV field")
    if use_flux and uv is None:
        raise ValueError("measure='flux' needs the (U, V) vapour-flux vectors")

    for k in range(T):
        pairs = overlap_pairs(labels[k - 1], labels[k], min_overlap=min_overlap,
                              dilate=dilate) if k else {}
        inbound: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for (pa, ch), n in pairs.items():
            inbound[int(ch)].append((int(pa), n))

        # Contributions are only needed at genuine merges, so the measure is
        # computed lazily: on most frames nothing is competing and the whole
        # `_component_measure` pass is skipped.
        need_measure = (mode == "dominance" and not use_flux and k and any(
            len({own[k - 1].get(p) for p, _ in ps if p in own[k - 1]}) >= 2
            for ps in inbound.values()))
        meas_prev = (D.component_measure(labels[k - 1], iwv[k - 1], area, measure)
                     if need_measure else {})

        fx: dict = {}
        if use_flux and k:
            U, V = uv
            fx = D.edge_fluxes(labels[k - 1], labels[k], U[k], V[k])
        by_child: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        amb_child: dict[int, float] = defaultdict(float)
        if use_flux:
            for (p, c), f in fx.items():
                if f <= 0:
                    continue
                if p == 0:
                    amb_child[c] += f
                else:
                    L = own[k - 1].get(int(p))
                    if L is not None:
                        by_child[c][L] += f

        merged_units = _merge_units(labels[k], component[k], inbound, own[k - 1]) \
            if (only_at_merge and component is not None and k) else None

        for b in np.unique(labels[k]):
            b = int(b)
            if b == 0:
                continue
            reg = labels[k] == b
            seeded = bool((marker[k] & reg).any())

            ps = inbound.get(b, [])
            plins = sorted({own[k - 1][p] for p, _ in ps if p in own[k - 1]}) \
                if k else []

            if mode == "reachability":
                if seeded or TC in plins:
                    own[k][b] = TC                       # re-seed / continuation
                elif plins:
                    own[k][b] = plins[0]                 # oldest surviving parent
                else:
                    own[k][b] = next_lin
                    events.append(dict(frame=k, event="birth", lineage=next_lin))
                    next_lin += 1
                if len(plins) > 1:
                    for j in plins:
                        if j != own[k][b]:
                            events.append(dict(frame=k, event="merge",
                                               lineage=int(j), into=int(own[k][b])))
                continue

            # ---- dominance ----
            if merged_units is not None and not merged_units.get(b, True):
                # Not at a genuine merge: behave as the component does, i.e.
                # continue the sole parent lineage rather than subdividing.
                if seeded or TC in plins:
                    own[k][b] = TC
                elif plins:
                    own[k][b] = plins[0]
                else:
                    own[k][b] = next_lin
                    events.append(dict(frame=k, event="birth", lineage=next_lin))
                    next_lin += 1
                continue

            if use_flux:
                contrib = dict(by_child.get(b, {}))
                ambient = amb_child.get(b, 0.0)
                owner, _share, den = D.resolve_owner(contrib, frac, ambient,
                                                     include_ambient=True)
            else:
                contrib = defaultdict(float)
                for p, _n in ps:
                    L = own[k - 1].get(p)
                    if L is not None:
                        contrib[L] += meas_prev.get(p, 0.0)
                contrib = dict(contrib)
                ambient = 0.0
                owner, _share, den = D.resolve_owner(contrib, frac)

            if seeded or owner == TC:
                own[k][b] = TC
            elif owner is not None:
                own[k][b] = owner
            else:
                own[k][b] = next_lin
                events.append(dict(frame=k, event="birth", lineage=next_lin))
                next_lin += 1

            if own[k][b] == TC and len(contrib) >= 2:
                for L in contrib:
                    if L != TC:
                        events.append(dict(frame=k, event="merge", lineage=int(L),
                                           into=TC))
            if use_flux and own[k][b] == TC and den > 0:
                amb_num += ambient
                amb_den += den

        if k:
            fan: dict[int, list[int]] = defaultdict(list)
            for (pa, ch) in pairs:
                fan[pa].append(ch)
            for pa, chs in fan.items():
                if len(chs) > 1 and pa in own[k - 1]:
                    events.append(dict(frame=k, event="split",
                                       lineage=int(own[k - 1][pa]),
                                       n_children=len(chs)))
            alive = {c for (_, c) in pairs}
            for pb, pl in own[k - 1].items():
                if pb not in alive:
                    events.append(dict(frame=k, event="death", lineage=int(pl)))

    lineage = _paint(labels, own)
    ambient_share = float(amb_num / amb_den) if (use_flux and amb_den > 0) else float("nan")
    cols = ["frame", "event", "lineage", "into", "n_children"]
    ev = pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
    return dict(lineage=lineage, events=ev, n_lineages=int(next_lin - TC - 1),
                ambient_share=ambient_share)


def _merge_units(lab2d, comp2d, inbound, own_prev) -> dict[int, bool]:
    """basin -> is its COMPONENT at a genuine merge (>=2 inbound lineages)?

    Implements `only_at_merge`.  Pooled at the component so that when a front
    reaches a TC, every basin of the joined blob is arbitrated, not only the
    basins that happen to touch the interface.
    """
    by_comp: dict[int, set] = defaultdict(set)
    basins_of: dict[int, list[int]] = defaultdict(list)
    for b in np.unique(lab2d):
        b = int(b)
        if b == 0:
            continue
        c = int(np.bincount(comp2d[lab2d == b].ravel()).argmax())
        basins_of[c].append(b)
        for p, _n in inbound.get(b, []):
            L = own_prev.get(p)
            if L is not None:
                by_comp[c].add(L)
    out: dict[int, bool] = {}
    for c, bs in basins_of.items():
        at_merge = len(by_comp.get(c, ())) >= 2
        for b in bs:
            out[b] = at_merge
    return out


def _paint(labels: np.ndarray, own: list[dict[int, int]]) -> np.ndarray:
    """Object-level ownership -> a per-cell int16 lineage field."""
    out = np.zeros(labels.shape, np.int16)
    for k in range(labels.shape[0]):
        if not own[k]:
            continue
        lut = np.zeros(int(labels[k].max()) + 1, np.int16)
        for b, l in own[k].items():
            lut[b] = l
        out[k] = lut[labels[k]]
    return out


def _undirected(labels, marker, min_overlap, dilate) -> dict:
    """ABLATION: undirected reachability.  Edges both ways; BFS floods backward.

    Kept so the causal fix is a measured delta rather than an assertion.
    Everything the TC ever touches becomes the TC, from frame 0.
    """
    T = labels.shape[0]
    edges = defaultdict(set)
    for k in range(T - 1):
        for (a, b) in overlap_pairs(labels[k], labels[k + 1],
                                    min_overlap=min_overlap, dilate=dilate):
            edges[(k, a)].add((k + 1, b))
            edges[(k + 1, b)].add((k, a))              # <- the bug, made explicit
    seeds = {(k, int(b)) for k in range(T)
             for b in np.unique(labels[k][marker[k]]) if b > 0}
    seen, q = set(seeds), list(seeds)
    while q:
        for nb in edges[q.pop()]:
            if nb not in seen:
                seen.add(nb)
                q.append(nb)
    lineage = np.zeros(labels.shape, np.int16)
    for k in range(T):
        for b in np.unique(labels[k]):
            if b > 0:
                lineage[k][labels[k] == b] = TC if (k, int(b)) in seen else TC + 1
    return dict(lineage=lineage, events=pd.DataFrame(
        columns=["frame", "event", "lineage", "into", "n_children"]),
        n_lineages=1, ambient_share=float("nan"))


# ══════════════════════════════════════════════════════════════════════════════
def build_variant(variant: str, seg: dict, marker: np.ndarray, *,
                  measure: str = "content", frac: float = D.MAIN_FEEDER_FRAC,
                  iwv: np.ndarray | None = None, uv: tuple | None = None,
                  area: np.ndarray | None = None, only_at_merge: bool = True,
                  min_overlap: int = 1, dilate: int = 1) -> dict:
    """`build` with the label field and mode chosen by variant name.

    `seg` is the dict from the segment stage: {"component", "basin", ...}.
    """
    if variant not in VARIANT_SPEC:
        raise KeyError(f"unknown variant {variant!r}; "
                       f"known: {sorted(VARIANT_SPEC)}")
    mode, gran = VARIANT_SPEC[variant]
    labels = seg["basin"] if gran == "basin" else seg["component"]
    res = build(labels, marker, mode=mode, measure=measure, frac=frac,
                iwv=iwv, uv=uv, area=area, component=seg["component"],
                only_at_merge=only_at_merge and gran == "basin",
                min_overlap=min_overlap, dilate=dilate)
    res["variant"] = variant
    res["mode"] = mode
    res["granularity"] = gran
    return res


def corridor_of(lineage: np.ndarray) -> np.ndarray:
    """The TC's cells."""
    return np.asarray(lineage) == TC


def area_stats(lineage: np.ndarray, area_km2: np.ndarray) -> dict:
    """Corridor extent diagnostics for the index row."""
    cor = corridor_of(lineage)
    per_frame = np.array([(area_km2 * cor[k]).sum() for k in range(cor.shape[0])])
    box = float(area_km2.sum())
    return dict(area_km2=per_frame, mean_area_km2=float(np.nanmean(per_frame)),
                area_frac_box=float(np.nanmean(per_frame) / box) if box else 0.0,
                box_area_km2=box, total_area_km2=float(per_frame.sum()),
                seeded_frames=int(cor.reshape(cor.shape[0], -1).any(1).sum()))


def pruned_fraction(lineage_all: np.ndarray, lineage_sub: np.ndarray,
                    area_km2: np.ndarray) -> dict:
    """How much of the all-interactions corridor a subset variant removed.

    `lineage_sub` must be a subset variant (`stem`, `stem_ws`, `excl`...).  The
    subset property is asserted rather than assumed, because it is what the
    pruned link tables and the exclusive re-attribution both rely on: if it ever
    fails, a hazard could be associated under the main stem and not under
    all-interactions, and the delta would stop being interpretable.
    """
    a = corridor_of(lineage_all)
    b = corridor_of(lineage_sub)
    leak = int((b & ~a).sum())
    pruned = a & ~b
    A = area_km2[None, :, :]
    tot = float((A * a).sum())
    return dict(pruned_area_km2=float((A * pruned).sum()),
                pruned_frac=float((A * pruned).sum() / tot) if tot > 0 else 0.0,
                subset_violations=leak)
