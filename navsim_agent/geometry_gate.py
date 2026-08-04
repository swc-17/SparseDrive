"""Fail-closed geometry source gate for the ego planner."""

from collections.abc import Mapping

import torch


TEACHER_DETECTIONS_EMPERROR_MAPS = "teacher_detections_emperror_maps"
EMPERROR_DETECTIONS_TEACHER_MAPS = "emperror_detections_teacher_maps"
GENERATED_GEOMETRY_SOURCES = (
    "emperror",
    TEACHER_DETECTIONS_EMPERROR_MAPS,
    EMPERROR_DETECTIONS_TEACHER_MAPS,
)
PLANNER_GEOMETRY_SOURCES = ("teacher", *GENERATED_GEOMETRY_SOURCES)


def planner_geometry(
    source,
    teacher_detections,
    teacher_maps,
    generated: Mapping[str, torch.Tensor] | None = None,
):
    """Select and validate one dense top-50/top-10 geometry source."""

    if source in GENERATED_GEOMETRY_SOURCES:
        if generated is None:
            raise RuntimeError("EMPERROR geometry source requires frame output")
        if not isinstance(generated, Mapping):
            raise TypeError("EMPERROR frame output must be a mapping")
        generated_detections = generated["anchor_bbox"]
        generated_maps = generated["map_anchor"]
        detection_valid = generated["detection_valid"]
        map_valid = generated["map_valid"]

    if source == "teacher":
        detections, maps = teacher_detections, teacher_maps
    elif source == "emperror":
        detections, maps = generated_detections, generated_maps
    elif source == TEACHER_DETECTIONS_EMPERROR_MAPS:
        detections, maps = teacher_detections, generated_maps
    elif source == EMPERROR_DETECTIONS_TEACHER_MAPS:
        detections, maps = generated_detections, teacher_maps
    else:
        raise ValueError(f"unknown planner geometry source {source!r}")

    expected_detections = (teacher_detections.shape[0], 50, 11)
    expected_maps = (teacher_detections.shape[0], 10, 40)
    if detections.shape != expected_detections or maps.shape != expected_maps:
        raise ValueError(
            "planner geometry must be [B,50,11] detections and [B,10,40] maps"
        )
    if not detections.is_floating_point() or not maps.is_floating_point():
        raise TypeError("planner geometry must be floating point")
    if source in GENERATED_GEOMETRY_SOURCES:
        if detection_valid.shape != expected_detections[:2]:
            raise ValueError("detection validity mask must be [B,50]")
        if map_valid.shape != expected_maps[:2]:
            raise ValueError("map validity mask must be [B,10]")
        if detection_valid.dtype != torch.bool or map_valid.dtype != torch.bool:
            raise TypeError("planner geometry validity masks must be boolean")
        if not bool(detection_valid.all()) or not bool(map_valid.all()):
            raise ValueError("planner geometry cannot contain padded slots")
    if (
        detections.device != teacher_detections.device
        or maps.device != teacher_maps.device
    ):
        raise ValueError("planner geometry must already be on the teacher tensors' device")
    if detections.dtype != teacher_detections.dtype or maps.dtype != teacher_maps.dtype:
        raise TypeError("planner geometry must match the teacher tensors' dtype")
    if not bool(torch.isfinite(detections).all()) or not bool(
        torch.isfinite(maps).all()
    ):
        raise ValueError("planner geometry must be finite")
    return detections, maps
