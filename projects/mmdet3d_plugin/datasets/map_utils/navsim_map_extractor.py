"""Offline nuPlan-GPKG -> SparseDrive local vector-map extractor (NAVSIM Phase 2).

Mirrors ``NuscMapExtractor`` semantics for the three SparseDrive map classes,
but reads the nuPlan map GPKGs that ship with NAVSIM instead of the nuScenes
map expansion:

- ``ped_crossing``: ``crosswalks`` polygons, close ones merged, emitted as
  closed contour polylines (same ``_union_ped``/contour logic as nuScenes);
- ``divider``: lane-boundary polylines from the ``boundaries`` layer that are
  *interior* lane separators. A boundary fid referenced by
  ``lanes_polygons.left/right_boundary_fid`` counts as a divider when it is
  shared by >= 2 lanes (adjacent lanes reference the same fid) or has
  ``boundary_type_fid == 0`` (painted dashed divider). Singly-referenced
  solid boundaries are road edges: they coincide with the drivable-area
  contour and are covered by the ``boundary`` class (verified empirically on
  us-nv-las-vegas-strip);
- ``boundary``: contour of the union of the drivable polygons
  (``road_segments`` + ``lanes_polygons`` + ``intersections`` +
  ``generic_drivable_areas`` + ``carpark_areas``), clockwise exteriors /
  counter-clockwise interiors exactly like the nuScenes extractor.

Coordinate contract: the GPKGs store EPSG:4326 geometry; each map's ``meta``
layer records the projected CRS (UTM) that nuPlan/NAVSIM ego poses live in.
All geometries are re-projected to that CRS once at load. Per query the
caller passes the **SparseDrive-frame** ego2global pose produced by
``navsim_converter`` (G_sd = G_nav @ inv(C4)); extracting the SE(2) yaw from
that pose and mapping global points with ``p_local = Rz(-yaw_sd) (p - t)``
lands geometry directly in the SparseDrive BEV frame (x right, y forward),
so no additional NAVSIM->SparseDrive rotation is needed here.

Only used offline by ``tools/data_converter/navsim_converter.py``; the
SparseDrive runtime consumes the resulting ``map_annos`` without importing
fiona/pyproj or any nuPlan SDK.
"""

import os
from collections import Counter

import numpy as np
from shapely import affinity, ops, strtree
from shapely.geometry import LineString, Polygon, box, shape

from .utils import (
    get_drivable_area_contour,
    get_ped_crossing_contour,
    split_collections,
)

MAP_LOCATIONS = (
    "sg-one-north",
    "us-ma-boston",
    "us-nv-las-vegas-strip",
    "us-pa-pittsburgh-hazelwood",
)

# drivable area = union of these polygon layers (nuPlan DRIVABLE_AREA
# definition; mirrors nuScenes road_segment + lane union).
DRIVABLE_LAYERS = (
    "road_segments",
    "lanes_polygons",
    "intersections",
    "generic_drivable_areas",
    "carpark_areas",
)


def _union_ped(ped_geoms):
    """Merge close, similarly-oriented ped crossings (copy of the nuScenes
    extractor logic, kept here so this module never imports the nuScenes
    map API)."""

    def get_rec_direction(geom):
        rect = geom.minimum_rotated_rectangle
        rect_v_p = np.array(rect.exterior.coords)[:3]
        rect_v = rect_v_p[1:] - rect_v_p[:-1]
        v_len = np.linalg.norm(rect_v, axis=-1)
        longest_v_i = v_len.argmax()
        return rect_v[longest_v_i], v_len[longest_v_i]

    tree = strtree.STRtree(ped_geoms)
    index_by_id = {id(pt): i for i, pt in enumerate(ped_geoms)}

    final_pgeom = []
    remain_idx = list(range(len(ped_geoms)))
    for i, pgeom in enumerate(ped_geoms):
        if i not in remain_idx:
            continue
        remain_idx.pop(remain_idx.index(i))
        pgeom_v, pgeom_v_norm = get_rec_direction(pgeom)
        final_pgeom.append(pgeom)
        for o in tree.query(pgeom):
            o_idx = index_by_id[id(o)]
            if o_idx not in remain_idx:
                continue
            o_v, o_v_norm = get_rec_direction(o)
            cos = pgeom_v.dot(o_v) / (pgeom_v_norm * o_v_norm)
            if 1 - np.abs(cos) < 0.01:  # theta < 8 degrees
                final_pgeom[-1] = final_pgeom[-1].union(o)
                remain_idx.pop(remain_idx.index(o_idx))

    results = []
    for p in final_pgeom:
        results.extend(split_collections(p))
    return results


class _LocationMap:
    """All per-location geometry, re-projected to the map's UTM CRS."""

    def __init__(self, gpkg_path):
        import fiona
        from pyproj import Transformer

        with fiona.open(gpkg_path, layer="meta") as src:
            meta = {f["properties"]["key"]: f["properties"]["value"]
                    for f in src}
        geo_crs = meta.get("geographicCoordSystem", "epsg:4326")
        proj_crs = meta["projectedCoordSystem"]
        tr = Transformer.from_crs(geo_crs, proj_crs, always_xy=True)

        def to_utm(geom_mapping):
            g = shape(geom_mapping)
            return ops.transform(
                lambda x, y, z=None: tr.transform(x, y), g
            )

        # lane -> boundary references over the FULL map (multiplicity must
        # be global; a patch-local count could miss the second lane).
        ref_count = Counter()
        with fiona.open(gpkg_path, layer="lanes_polygons") as src:
            for f in src:
                ref_count[int(f["properties"]["left_boundary_fid"])] += 1
                ref_count[int(f["properties"]["right_boundary_fid"])] += 1

        dividers = []
        with fiona.open(gpkg_path, layer="boundaries") as src:
            for f in src:
                fid = int(f.id)
                if fid not in ref_count:
                    continue
                is_divider = (
                    ref_count[fid] >= 2
                    or int(f["properties"]["boundary_type_fid"]) == 0
                )
                if not is_divider:
                    continue  # singly-referenced solid line = road edge
                g = to_utm(f["geometry"])
                if g.is_valid and not g.is_empty:
                    dividers.append(g)

        crosswalks = []
        with fiona.open(gpkg_path, layer="crosswalks") as src:
            for f in src:
                g = to_utm(f["geometry"]).buffer(0)
                if g.is_valid and not g.is_empty:
                    crosswalks.append(g)

        drivable = []
        for layer in DRIVABLE_LAYERS:
            with fiona.open(gpkg_path, layer=layer) as src:
                for f in src:
                    g = to_utm(f["geometry"]).buffer(0)
                    if g.is_valid and not g.is_empty:
                        drivable.append(g)

        self.divider_tree = strtree.STRtree(dividers)
        self.crosswalk_tree = strtree.STRtree(crosswalks)
        self.drivable_tree = strtree.STRtree(drivable)
        self.counts = dict(
            dividers=len(dividers),
            crosswalks=len(crosswalks),
            drivable=len(drivable),
        )


class NavsimMapExtractor(object):
    """NAVSIM/nuPlan local vector-map extractor.

    Args:
        maps_root (str): directory holding ``<location>/<version>/map.gpkg``.
        roi_size (tuple): BEV range (x extent, y extent) in the SparseDrive
            frame, e.g. (30, 60) = x in [-15, 15], y in [-30, 30].
    """

    def __init__(self, maps_root, roi_size):
        self.maps_root = maps_root
        self.roi_size = tuple(roi_size)
        self.local_patch = box(
            -roi_size[0] / 2, -roi_size[1] / 2,
            roi_size[0] / 2, roi_size[1] / 2,
        )
        self._maps = {}

    def _gpkg_path(self, location):
        loc_dir = os.path.join(self.maps_root, location)
        versions = sorted(
            d for d in os.listdir(loc_dir)
            if os.path.isdir(os.path.join(loc_dir, d))
        )
        assert len(versions) == 1, (
            f"expected exactly one map version in {loc_dir}, got {versions}"
        )
        return os.path.join(loc_dir, versions[0], "map.gpkg")

    def _get_map(self, location):
        assert location in MAP_LOCATIONS, f"unknown map location {location}"
        if location not in self._maps:
            path = self._gpkg_path(location)
            print(f"[NavsimMapExtractor] loading {path} ...")
            self._maps[location] = _LocationMap(path)
            print(f"[NavsimMapExtractor] {location}: "
                  f"{self._maps[location].counts}")
        return self._maps[location]

    def get_map_geom(self, location, translation, yaw_sd):
        """Extract local map geometry for one ego pose.

        Args:
            location (str): map location name.
            translation (array-like): global xy(z) of the SparseDrive-frame
                ego2global pose (identical to the NAVSIM global translation).
            yaw_sd (float): SE(2) yaw of the SparseDrive-frame ego2global
                rotation (atan2(R[1,0], R[0,0]) of G_sd).

        Returns:
            dict with LineString lists for ``ped_crossing`` / ``divider`` /
            ``boundary`` plus the clipped ``drivable_area`` polygons — the
            same shape ``NuscMapExtractor.get_map_geom`` returns; all
            geometry is in the SparseDrive local BEV frame.
        """
        loc_map = self._get_map(location)
        tx, ty = float(translation[0]), float(translation[1])

        # ROI patch in global coordinates (rotate local patch to ego yaw)
        patch_global = affinity.translate(
            affinity.rotate(self.local_patch, yaw_sd, origin=(0, 0),
                            use_radians=True),
            xoff=tx, yoff=ty,
        )

        def to_local(geom):
            g = affinity.translate(geom, xoff=-tx, yoff=-ty)
            return affinity.rotate(g, -yaw_sd, origin=(0, 0),
                                   use_radians=True)

        # dividers: clip lane-boundary lines to the local patch
        all_dividers = []
        for g in loc_map.divider_tree.query(patch_global):
            line = to_local(g).intersection(self.local_patch)
            if line.is_empty:
                continue
            for piece in split_collections(line):
                if piece.geom_type == "LineString" and piece.length > 0:
                    all_dividers.append(piece)

        # ped crossings: clip polygons, merge close ones, take contours
        ped_crossings = []
        for g in loc_map.crosswalk_tree.query(patch_global):
            poly = to_local(g).intersection(self.local_patch)
            if poly.is_empty:
                continue
            for piece in split_collections(poly):
                if piece.geom_type == "Polygon" and piece.area > 0:
                    ped_crossings.append(piece)
        if ped_crossings:
            ped_crossings = _union_ped(ped_crossings)
        ped_crossing_lines = []
        for p in ped_crossings:
            line = get_ped_crossing_contour(p, self.local_patch)
            if line is not None:
                ped_crossing_lines.append(line)

        # boundary: contour of the clipped drivable-area union
        clipped = []
        for g in loc_map.drivable_tree.query(patch_global):
            poly = to_local(g).intersection(self.local_patch)
            if not poly.is_empty:
                clipped.append(poly)
        drivable_areas = split_collections(ops.unary_union(clipped)) \
            if clipped else []
        drivable_areas = [p for p in drivable_areas
                          if p.geom_type == "Polygon" and p.area > 0]
        boundaries = get_drivable_area_contour(drivable_areas, self.roi_size)

        return dict(
            divider=all_dividers,           # List[LineString]
            ped_crossing=ped_crossing_lines,  # List[LineString]
            boundary=boundaries,            # List[LineString]
            drivable_area=drivable_areas,   # List[Polygon]
        )
