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
  counter-clockwise interiors exactly like the nuScenes extractor;
- ``stop_line``: traffic-light-controlled stop bars from the
  ``stop_polygons`` layer (``stop_polygon_type_fid == 2`` only). Each stop
  polygon is a thin quad spanning the lane width; the emitted polyline is
  the midline of its minimum rotated rectangle along the long axis (within
  half the bar thickness of the painted line). Each stop line carries
  aligned association extras (``stop_line_extras``): the mean position of
  its controlling traffic-light bulbs (``stop_polygons.traffic_light_fids``
  -> ``traffic_lights`` points) in the local frame, and the lane-connector
  fids referencing it via ``lane_connectors.traffic_light_stop_line_fids``
  (used by the converter to resolve per-frame red/green status).

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

# stop_polygons.stop_polygon_type_fid for traffic-light-controlled stop
# lines (nuplan StopLineType.TRAFFIC_LIGHT; 0=ped_crossing, 1=stop_sign,
# 3=turn_stop, 4=yield are intentionally excluded).
STOP_POLYGON_TYPE_TRAFFIC_LIGHT = 2


def _parse_fids(value):
    """Comma-separated GPKG fid-list string -> list of ints."""
    if value is None:
        return []
    return [int(tok) for tok in str(value).split(",") if tok.strip()]


def _stop_bar_line(poly):
    """Stop polygon -> stop-bar midline (LineString) or None.

    The midline of the minimum rotated rectangle along its long axis: the
    polygon spans the lane width with a thin (~0.5 m) extent along travel
    direction, so this lands on the painted bar to within half thickness.
    """
    rect = poly.minimum_rotated_rectangle
    if rect.geom_type != "Polygon":
        return None
    corners = np.array(rect.exterior.coords)[:4]
    e01 = np.linalg.norm(corners[1] - corners[0])
    e12 = np.linalg.norm(corners[2] - corners[1])
    if e01 >= e12:  # long axis along corners[0] -> corners[1]
        p0 = (corners[3] + corners[0]) / 2.0
        p1 = (corners[1] + corners[2]) / 2.0
    else:  # long axis along corners[1] -> corners[2]
        p0 = (corners[0] + corners[1]) / 2.0
        p1 = (corners[2] + corners[3]) / 2.0
    if np.linalg.norm(p1 - p0) <= 0:
        return None
    return LineString([p0, p1])


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
        for o_idx in tree.query(pgeom):
            o = ped_geoms[o_idx]
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

        # traffic-light bulb positions: fid -> UTM xy
        tl_xy = {}
        with fiona.open(gpkg_path, layer="traffic_lights") as src:
            for f in src:
                g = to_utm(f["geometry"])
                tl_xy[int(f.id)] = (g.x, g.y)

        # traffic-light-controlled stop polygons + their TL association
        stop_geoms, stop_meta = [], []
        with fiona.open(gpkg_path, layer="stop_polygons") as src:
            for f in src:
                type_fid = int(f["properties"]["stop_polygon_type_fid"])
                if type_fid != STOP_POLYGON_TYPE_TRAFFIC_LIGHT:
                    continue
                g = to_utm(f["geometry"]).buffer(0)
                if not g.is_valid or g.is_empty:
                    continue
                stop_geoms.append(g)
                stop_meta.append(dict(
                    fid=int(f.id),
                    tl_fids=_parse_fids(
                        f["properties"]["traffic_light_fids"]
                    ),
                ))

        # stop polygon fid -> lane connector fids referencing it (per-frame
        # TL status in the logs is keyed by lane connector id)
        stop_connectors = {}
        with fiona.open(gpkg_path, layer="lane_connectors") as src:
            for f in src:
                for sfid in _parse_fids(
                    f["properties"]["traffic_light_stop_line_fids"]
                ):
                    stop_connectors.setdefault(sfid, []).append(int(f.id))

        self.dividers = dividers
        self.crosswalks = crosswalks
        self.drivable = drivable
        self.stop_lines = stop_geoms
        self.divider_tree = strtree.STRtree(dividers)
        self.crosswalk_tree = strtree.STRtree(crosswalks)
        self.drivable_tree = strtree.STRtree(drivable)
        self.tl_xy = tl_xy
        self.stop_meta = stop_meta
        self.stop_tree = strtree.STRtree(stop_geoms)
        self.stop_index_by_id = {id(g): i for i, g in enumerate(stop_geoms)}
        self.stop_connectors = stop_connectors
        self.counts = dict(
            dividers=len(dividers),
            crosswalks=len(crosswalks),
            drivable=len(drivable),
            stop_lines=len(stop_geoms),
            traffic_lights=len(tl_xy),
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
            ``boundary`` / ``stop_line`` plus the clipped ``drivable_area``
            polygons — the same shape ``NuscMapExtractor.get_map_geom``
            returns; all geometry is in the SparseDrive local BEV frame.
            ``stop_line_extras`` is a list aligned with ``stop_line``:
            dict(tl_xy=(2,) float array or None, connector_fids=[int]).
        """
        loc_map = self._get_map(location)
        tx, ty = float(translation[0]), float(translation[1])
        cos_y, sin_y = np.cos(yaw_sd), np.sin(yaw_sd)

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
        for idx_d in loc_map.divider_tree.query(patch_global):
            g = loc_map.dividers[idx_d]
            line = to_local(g).intersection(self.local_patch)
            if line.is_empty:
                continue
            for piece in split_collections(line):
                if piece.geom_type == "LineString" and piece.length > 0:
                    all_dividers.append(piece)

        # ped crossings: clip polygons, merge close ones, take contours
        ped_crossings = []
        for idx_c in loc_map.crosswalk_tree.query(patch_global):
            g = loc_map.crosswalks[idx_c]
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
        for idx_v in loc_map.drivable_tree.query(patch_global):
            g = loc_map.drivable[idx_v]
            poly = to_local(g).intersection(self.local_patch)
            if not poly.is_empty:
                clipped.append(poly)
        drivable_areas = split_collections(ops.unary_union(clipped)) \
            if clipped else []
        drivable_areas = [p for p in drivable_areas
                          if p.geom_type == "Polygon" and p.area > 0]
        boundaries = get_drivable_area_contour(drivable_areas, self.roi_size)

        # stop lines: TL-controlled stop-bar midlines clipped to the patch,
        # with aligned TL-association extras per emitted piece
        stop_lines, stop_line_extras = [], []
        for idx_s in loc_map.stop_tree.query(patch_global):
            g = loc_map.stop_lines[idx_s]
            idx = idx_s
            meta = loc_map.stop_meta[idx]
            bar = _stop_bar_line(g)
            if bar is None:
                continue
            line = to_local(bar).intersection(self.local_patch)
            if line.is_empty:
                continue
            tl_pts = [
                loc_map.tl_xy[f] for f in meta["tl_fids"]
                if f in loc_map.tl_xy
            ]
            if tl_pts:
                # p_local = Rz(-yaw_sd) @ (p_global - t); the mean TL bulb
                # position may lie outside the ROI patch (that is fine)
                mean = np.asarray(tl_pts, dtype=np.float64).mean(axis=0)
                dx, dy = mean[0] - tx, mean[1] - ty
                tl_local = np.array(
                    [cos_y * dx + sin_y * dy, -sin_y * dx + cos_y * dy]
                )
            else:
                tl_local = None
            connector_fids = loc_map.stop_connectors.get(meta["fid"], [])
            for piece in split_collections(line):
                if piece.geom_type == "LineString" and piece.length > 0:
                    stop_lines.append(piece)
                    stop_line_extras.append(dict(
                        tl_xy=tl_local,
                        connector_fids=connector_fids,
                    ))

        return dict(
            divider=all_dividers,           # List[LineString]
            ped_crossing=ped_crossing_lines,  # List[LineString]
            boundary=boundaries,            # List[LineString]
            stop_line=stop_lines,           # List[LineString]
            stop_line_extras=stop_line_extras,  # aligned with stop_line
            drivable_area=drivable_areas,   # List[Polygon]
        )
