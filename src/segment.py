"""Stage 2a -- watershed basins with PERSISTENT labels.  The new piece.

WHAT THIS ADDS
    `tcmc.corridor.thr.components` gives per-frame connected components.  A
    component that is really two moisture plumes joined at a neck is one label,
    and the ownership decision for it is therefore all-or-nothing: at a merge
    the whole blob goes to the dominant feeder, or none of it does.  This module
    subdivides each component into watershed basins and gives each basin an
    identity that persists across frames, so the downstream question becomes
    "which basins does the TC dominantly feed" rather than "does the TC keep the
    blob".

THE CONSTRUCTION, AND WHERE IT COMES FROM
    Segmentation and tracking happen in ONE forward pass, after

        Pinidiyaarachchi, A. and Wahlby, C. (2005): Seeded Watersheds for
        Combined Segmentation and Tracking of Cells.  ICIAP 2005, LNCS 3617,
        336-343.  doi:10.1007/11553595_41

    whose construction is exactly this: the previous frame's segmentation
    supplies seeds for the current frame's watershed, those seeds are combined
    with morphological seeds from the current frame, and rule-based merging
    reduces over-segmentation while propagating labels from one frame to the
    next.  The consequence here is that the `basin` label IS the branch id.  The
    lineage stage then only has to assign OWNERSHIP to an identity that already
    exists, which is what keeps segmentation independent of the variant.

    Per frame:
      1. morphological seeds   h-maxima of the THR anomaly inside the mask.
                               `h` is the dynamics threshold: a maximum survives
                               only if it stands `h` above the saddle joining it
                               to a higher one.  This is the ONE knob, and it is
                               a continuous dial -- h -> inf gives one basin per
                               component and reproduces the non-watershed
                               pipeline bit for bit, h -> 0 gives a basin per
                               local maximum.
      2. identity              each seed region is matched to the previous
                               frame's basins by dilated overlap.  One parent ->
                               continuation, keep the id.  Two or more -> MERGE,
                               take the largest-overlap parent's id and record
                               the others.  None -> BIRTH.  Two seeds sharing one
                               parent -> SPLIT, the larger keeps the id.
      3. persistence           with `seed_prev_frame`, a previous basin whose
                               maximum has weakened below `h` but which still has
                               distinct area is re-seeded from its own overlap
                               region rather than being absorbed silently.
      4. flooding              `skimage.segmentation.watershed` on the negated
                               anomaly (basins are moisture maxima), or on the
                               Sobel gradient with `edge_based=True`, after
                               Jones, Christensen and Stier (2023), AMT 16:1043,
                               Sect. 3, whose anvil detection floods a gradient
                               field precisely to avoid a fixed threshold.
      5. tidying               basins below `min_basin_km2` are merged into the
                               neighbour across their longest shared border,
                               which is Pinidiyaarachchi and Wahlby's first
                               merging rule; and a basin that has become
                               spatially disconnected is split, the largest piece
                               keeping the id.

THE CONTAINMENT GUARANTEE
    The flood is constrained by `mask=mask[k]`, so no basin crosses a component
    boundary and no masked cell is left unassigned.  Basins therefore EXACTLY
    refine components.  Three things follow, and all three matter:
      * `h -> inf` is bit-identical to the current pipeline, so the watershed is
        a sensitivity axis rather than a new method;
      * the watershed corridor stays a subset of the all-interactions corridor,
        so the pruned-set logic and the exclusive pass remain valid;
      * a per-basin statistic sums to the per-component statistic, so the branch
        aggregates and the main-stem aggregate are consistent by construction.

WHAT IS NOT DONE HERE
    Ownership.  A basin is an object, not a TC parcel.  `only_at_merge` lives in
    the config beside `h` but is consumed by `tcmc.corridor.lineage`, because
    whether a component is "at a merge" is a fact about the PREVIOUS frame's
    ownership and segmentation does not know it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from tcmc.corridor.thr import ST, disk

try:                                                    # skimage is required, but
    from skimage.morphology import h_maxima as _h_maxima      # keep the import
    from skimage.segmentation import watershed as _watershed  # failure legible
    from skimage.filters import sobel as _sobel
except ImportError as _e:                                      # pragma: no cover
    raise ImportError("tcmc.corridor.segment needs scikit-image "
                      "(h_maxima, watershed, sobel)") from _e


# ══════════════════════════════════════════════════════════════════════════════
# Overlap, shared with the lineage sweep
# ══════════════════════════════════════════════════════════════════════════════
def overlap_pairs(a: np.ndarray, b: np.ndarray, *, min_overlap: int = 1,
                  dilate: int = 1) -> dict[tuple[int, int], int]:
    """{(label in a, label in b): overlapping cells}, after dilating `a`.

    The dilation absorbs the O(1)-cell boundary motion of independently
    thresholded frames.  It is applied to the BINARY support and the labels are
    re-read by nearest-label expansion, so no label is invented adjacent to a
    component that does not own it.

    Linking is deliberately PERMISSIVE.  Under-linking truncates the remnant
    tail and is unrecoverable; over-linking declares a merge, which is recorded
    and can be examined.
    """
    if dilate > 0:
        da = ndi.grey_dilation(a, footprint=disk(dilate))
        da = np.where(a > 0, a, da)                # never overwrite a real label
    else:
        da = a
    both = (da > 0) & (b > 0)
    if not both.any():
        return {}
    pr, cnt = np.unique(np.stack([da[both], b[both]], 1), axis=0, return_counts=True)
    return {(int(x), int(y)): int(c) for (x, y), c in zip(pr, cnt) if c >= min_overlap}


# ══════════════════════════════════════════════════════════════════════════════
# Seeds
# ══════════════════════════════════════════════════════════════════════════════
def _seed_regions(anom2d: np.ndarray, mask2d: np.ndarray, comp2d: np.ndarray,
                  prev_basin: np.ndarray | None, h: float, enabled: bool,
                  hysteresis: float) -> np.ndarray:
    """Frame-local seed regions, labelled 1..n.  One per intended basin.

    `h` alone decides HOW MANY basins a component has.  That is the whole
    content of the knob, and it is why a previously distinct parcel is not
    simply carried forward: if the saddle joining two parcels has become
    shallower than `h`, they are one parcel now, and re-seeding them separately
    would mean `h` controlled nothing.

    HYSTERESIS is how a parcel that merely weakens is nonetheless kept.  A
    basin that already exists survives on a lower bar, `h * hysteresis`, so a
    peak that dips for one frame is not absorbed and then reborn with a new id
    -- which would turn one branch into three and corrupt every lifetime
    statistic.  A NEW basin still has to clear the full `h`.  This is the
    standard two-threshold treatment, and unlike unconditional re-seeding it
    leaves `h` in charge: a genuine merge, where the saddle falls below even the
    lower bar, is still declared.

    With `enabled=False` the seed is the component itself, so the pipeline
    degenerates to connected components with persistent ids -- the control run,
    and the reason the non-watershed variants share this code path.
    """
    if not enabled:
        return comp2d.astype(np.int32)

    a = np.where(mask2d, anom2d.astype(np.float64), 0.0)

    def hm(level: float) -> np.ndarray:
        try:
            return (_h_maxima(a, float(level)) > 0) & mask2d
        except Exception:                                      # noqa: BLE001
            return np.zeros_like(mask2d)

    seeds_bin = hm(h)
    if prev_basin is not None and 0.0 < hysteresis < 1.0:
        seeds_bin |= hm(h * hysteresis) & (prev_basin > 0)

    seeds, _ = ndi.label(seeds_bin, structure=ST)
    seeds = seeds.astype(np.int32)

    # A component with no surviving maximum would be flooded from nothing and
    # dropped.  Give it one marker at its own argmax: `h` is meant to control
    # SUBDIVISION, never to delete a detected system.
    nxt = int(seeds.max()) + 1
    for c in np.unique(comp2d):
        c = int(c)
        if c == 0:
            continue
        reg = comp2d == c
        if seeds[reg].any():
            continue
        flat = np.where(reg, anom2d, -np.inf)
        y, x = np.unravel_index(int(np.argmax(flat)), flat.shape)
        seeds[y, x] = nxt
        nxt += 1
    return seeds


def _name_by_identity(prov: np.ndarray, prev_basin: np.ndarray | None,
                      next_id: int, min_overlap: int, dilate: int):
    """Give each FLOODED basin the id of the previous basin it continues.

    Naming happens after the flood, not on the markers.  A marker can be a
    single cell -- the per-component fallback above produces exactly that -- and
    matching a one-cell marker against the previous frame finds whichever basin
    happens to contain that cell, missing every other parent and therefore
    missing the merge.  The flooded region is the object; it is what should be
    matched.

    The rules are Pinidiyaarachchi and Wahlby's rule-based merging, stated in
    terms of overlap rather than border strength because the corridor links by
    overlap everywhere else and mixing two association criteria would make the
    event taxonomy incoherent:

        one parent, uncontested   -> CONTINUATION, keep the id
        two or more parents       -> MERGE, take the largest-overlap parent's
                                     id and record the others as absorbed
        a parent with two children-> SPLIT, the larger keeps the id
        no parent                 -> BIRTH
    """
    events: list[dict] = []
    out = np.zeros_like(prov, np.int32)
    labels = [int(v) for v in np.unique(prov) if v > 0]
    if not labels:
        return out, next_id, events

    if prev_basin is None:
        for s in labels:
            out[prov == s] = next_id
            events.append(dict(event="birth", basin=next_id))
            next_id += 1
        return out, next_id, events

    ov = overlap_pairs(prev_basin, prov, min_overlap=min_overlap, dilate=dilate)
    parents: dict[int, list[tuple[int, int]]] = {}
    for (p, s), n in ov.items():
        parents.setdefault(int(s), []).append((int(p), n))

    # Resolve claims before assigning, so the result is a function of the whole
    # frame rather than of iteration order.
    best_child: dict[int, tuple[int, int]] = {}
    for s, ps in parents.items():
        for p, n in ps:
            if p not in best_child or n > best_child[p][1]:
                best_child[p] = (s, n)

    for s in labels:
        ps = sorted(parents.get(s, []), key=lambda t: -t[1])
        claimable = [(p, n) for p, n in ps if best_child.get(p, (None,))[0] == s]
        if claimable:
            keep = claimable[0][0]
            out[prov == s] = keep
            for p, _n in ps:
                if p != keep:
                    events.append(dict(event="merge", basin=int(p), into=int(keep)))
        elif ps:
            out[prov == s] = next_id
            events.append(dict(event="split", basin=next_id, parent=int(ps[0][0])))
            next_id += 1
        else:
            out[prov == s] = next_id
            events.append(dict(event="birth", basin=next_id))
            next_id += 1
    return out, next_id, events


# ══════════════════════════════════════════════════════════════════════════════
# Tidying
# ══════════════════════════════════════════════════════════════════════════════
def _border_lengths(lab: np.ndarray, target: int) -> dict[int, int]:
    """Shared-border cell counts between basin `target` and each neighbour."""
    m = lab == target
    ring = ndi.binary_dilation(m, structure=ST) & ~m
    vals = lab[ring]
    vals = vals[vals > 0]
    if not vals.size:
        return {}
    u, c = np.unique(vals, return_counts=True)
    return {int(a): int(b) for a, b in zip(u, c)}


def _prune_small(lab: np.ndarray, area: np.ndarray, min_area_km2: float):
    """Merge basins below `min_area_km2` into the neighbour across the longest border.

    Pinidiyaarachchi and Wahlby's first merging step, with the border measured by
    length rather than by intensity: a corridor basin is a mass of water vapour,
    so a spurious sliver should join whichever real parcel it is most attached
    to.  A basin with no neighbour (a genuinely small isolated component) is
    left alone -- `thr.detect` has already applied the area filter that decides
    whether a SYSTEM is real, and this must not delete one.
    """
    if min_area_km2 <= 0:
        return lab, 0
    merged = 0
    for _ in range(8):                                # bounded: cascades terminate
        ids = [int(v) for v in np.unique(lab) if v > 0]
        if len(ids) <= 1:
            break
        areas = {i: float(area[lab == i].sum()) for i in ids}
        small = [i for i in ids if areas[i] < min_area_km2]
        if not small:
            break
        did = False
        for i in sorted(small, key=lambda x: areas[x]):
            nb = _border_lengths(lab, i)
            nb.pop(i, None)
            if not nb:
                continue
            into = max(nb, key=lambda k: (nb[k], areas.get(k, 0.0)))
            lab = np.where(lab == i, into, lab)
            merged += 1
            did = True
        if not did:
            break
    return lab, merged


def _split_disconnected(lab: np.ndarray, next_id: int):
    """A basin that has become two disjoint pieces keeps its id on the larger.

    Watershed with a single marker can produce a disconnected basin when the
    mask itself is disconnected under that marker's influence zone.  Leaving it
    would make "the branch's area" the sum of two unrelated parcels.
    """
    events: list[dict] = []
    for i in [int(v) for v in np.unique(lab) if v > 0]:
        piece, n = ndi.label(lab == i, structure=ST)
        if n <= 1:
            continue
        sizes = ndi.sum_labels(np.ones_like(piece, float), piece,
                               index=np.arange(1, n + 1))
        keep = 1 + int(np.argmax(sizes))
        for q in range(1, n + 1):
            if q == keep:
                continue
            lab = np.where(piece == q, next_id, lab)
            events.append(dict(event="split", basin=next_id, parent=i))
            next_id += 1
    return lab, next_id, events


# ══════════════════════════════════════════════════════════════════════════════
# The forward pass
# ══════════════════════════════════════════════════════════════════════════════
def basins(anom: np.ndarray, mask: np.ndarray, component: np.ndarray,
           area_km2: np.ndarray, *, enabled: bool = True, h: float = 25.0,
           min_basin_km2: float = 2.5e4, edge_based: bool = False,
           compactness: float = 0.0, seed_prev_frame: bool = True,
           hysteresis: float = 0.5, min_overlap: int = 1,
           dilate: int = 1) -> dict:
    """Persistent watershed basins for one storm.

    Order is SEED -> FLOOD -> TIDY -> NAME.  Naming last is what makes identity
    robust: the object matched against the previous frame is the flooded basin,
    not the marker that generated it.

    Returns
    -------
    basin    (T, Y, X) int32, 0 = background.  Labels are unique across the
             whole storm and persist across frames, so `basin` IS the branch id.
    events   DataFrame of birth / split / merge / death, at basin level
    n_basins number of distinct ids
    stats    dict: basins per frame, basins per component, merges, splits
    """
    T = mask.shape[0]
    out = np.zeros(mask.shape, np.int32)
    events: list[dict] = []
    next_id = 1
    prev = None
    n_pruned = 0
    hy = float(hysteresis) if seed_prev_frame else 1.0

    for k in range(T):
        if not mask[k].any():
            if prev is not None:
                for p in [int(v) for v in np.unique(prev) if v > 0]:
                    events.append(dict(frame=k, event="death", basin=p))
            prev = None
            continue

        seeds = _seed_regions(anom[k], mask[k], component[k], prev, h, enabled, hy)

        surface = _sobel(np.where(mask[k], anom[k], 0.0).astype(np.float64)) \
            if edge_based else -np.asarray(anom[k], np.float64)
        prov = _watershed(surface, markers=seeds, mask=mask[k], connectivity=1,
                          compactness=float(compactness),
                          watershed_line=False).astype(np.int32)

        prov, m = _prune_small(prov, area_km2, min_basin_km2)
        n_pruned += m
        prov, _n, _ev = _split_disconnected(prov, int(prov.max()) + 1)

        lab, next_id, ev = _name_by_identity(prov, prev, next_id, min_overlap,
                                             dilate)
        for e in ev:
            events.append(dict(frame=k, **e))

        if prev is not None:
            alive = {int(v) for v in np.unique(lab) if v > 0}
            absorbed = {int(e["basin"]) for e in ev if e["event"] == "merge"}
            for p in [int(v) for v in np.unique(prev) if v > 0]:
                if p not in alive and p not in absorbed:
                    events.append(dict(frame=k, event="death", basin=p))

        out[k] = lab
        prev = lab

    cols = ["frame", "event", "basin", "into", "parent"]
    ev_df = pd.DataFrame(events, columns=cols) if events \
        else pd.DataFrame(columns=cols)

    per_frame = np.array([len(np.unique(out[k])) - (1 if (out[k] == 0).any() else 0)
                          for k in range(T)], float)
    n_comp = np.array([len(np.unique(component[k]))
                       - (1 if (component[k] == 0).any() else 0)
                       for k in range(T)], float)
    live = n_comp > 0
    stats = dict(
        n_basins=int(len(np.unique(out)) - (1 if (out == 0).any() else 0)),
        basins_per_frame=float(per_frame[live].mean()) if live.any() else 0.0,
        components_per_frame=float(n_comp[live].mean()) if live.any() else 0.0,
        basins_per_component=float((per_frame[live] / n_comp[live]).mean())
        if live.any() else 1.0,
        n_pruned_small=int(n_pruned),
        n_merge_events=int((ev_df["event"] == "merge").sum()) if len(ev_df) else 0,
        n_split_events=int((ev_df["event"] == "split").sum()) if len(ev_df) else 0,
    )
    return dict(basin=out, events=ev_df, stats=stats, n_basins=stats["n_basins"])


def subdivided_components(component: np.ndarray, basin: np.ndarray) -> np.ndarray:
    """(T,) count of components that a watershed actually split into 2+ basins.

    The honest way to report how much work `h` is doing: a run where this is
    zero everywhere has produced results identical to the non-watershed
    pipeline, whatever `h` was set to.
    """
    T = component.shape[0]
    out = np.zeros(T, np.int32)
    for k in range(T):
        for c in np.unique(component[k]):
            if c == 0:
                continue
            n = len(np.unique(basin[k][component[k] == c]))
            if n > 1:
                out[k] += 1
    return out
