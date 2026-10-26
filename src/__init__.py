"""Corridor detection, segmentation, ownership and arbitration.

Pipeline within the subpackage:

    thr        |IVT| -> anomaly -> mask -> connected components
    marker     track -> per-frame centres, marker disc, C_link diagnostic
    segment    anomaly + mask -> watershed basins with PERSISTENT labels
    dominance  the four merge-arbitration measures
    lineage    forward-directed ownership; four variants, two granularities
    events     tau, junctions, pre-tau misattribution, the branch table
    arbitrate  cross-storm sequential arbitration -> the exclusive variants

Everything here is a function of the stored `fields/{sid}.nc` plus the track, so
the whole subpackage runs offline once stage 1 has been done once.
"""
from tcmc.corridor import (  # noqa: F401
    arbitrate, dominance, events, lineage, marker, segment, thr,
)
from tcmc.corridor.lineage import TC, VARIANT_SPEC  # noqa: F401

__all__ = ["thr", "marker", "segment", "dominance", "lineage", "events",
           "arbitrate", "TC", "VARIANT_SPEC"]
