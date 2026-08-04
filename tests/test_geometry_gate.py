import unittest

import torch

from navsim_agent.geometry_gate import (
    EMPERROR_DETECTIONS_TEACHER_MAPS,
    TEACHER_DETECTIONS_EMPERROR_MAPS,
    planner_geometry,
)


class GeometryGateTest(unittest.TestCase):
    def setUp(self):
        self.detections = torch.zeros(2, 50, 11)
        self.maps = torch.zeros(2, 10, 40)
        self.generated = {
            "anchor_bbox": self.detections + 1,
            "map_anchor": self.maps + 2,
            "detection_valid": torch.ones(2, 50, dtype=torch.bool),
            "map_valid": torch.ones(2, 10, dtype=torch.bool),
        }

    def test_routes_teacher_and_emperror(self):
        detections, maps = planner_geometry(
            "teacher", self.detections, self.maps, self.generated
        )
        self.assertIs(detections, self.detections)
        self.assertIs(maps, self.maps)

        detections, maps = planner_geometry(
            "emperror", self.detections, self.maps, self.generated
        )
        self.assertTrue(torch.equal(detections, self.detections + 1))
        self.assertTrue(torch.equal(maps, self.maps + 2))

    def test_routes_hybrid_sources(self):
        detections, maps = planner_geometry(
            TEACHER_DETECTIONS_EMPERROR_MAPS,
            self.detections,
            self.maps,
            self.generated,
        )
        self.assertIs(detections, self.detections)
        self.assertTrue(torch.equal(maps, self.maps + 2))

        detections, maps = planner_geometry(
            EMPERROR_DETECTIONS_TEACHER_MAPS,
            self.detections,
            self.maps,
            self.generated,
        )
        self.assertTrue(torch.equal(detections, self.detections + 1))
        self.assertIs(maps, self.maps)

    def test_invalid_external_geometry_fails_closed(self):
        invalid = (
            {**self.generated, "anchor_bbox": self.detections[:, :49]},
            {**self.generated, "map_anchor": self.maps[:, :, :39]},
            {
                **self.generated,
                "anchor_bbox": self.detections.clone().fill_(float("nan")),
            },
            {
                **self.generated,
                "detection_valid": torch.zeros(2, 50, dtype=torch.bool),
            },
        )
        for value in invalid:
            with self.assertRaises(ValueError):
                planner_geometry(
                    "emperror", self.detections, self.maps, value
                )

        with self.assertRaises(ValueError):
            planner_geometry("unknown", self.detections, self.maps)
        with self.assertRaises(RuntimeError):
            planner_geometry("emperror", self.detections, self.maps)
        with self.assertRaises(TypeError):
            planner_geometry(
                "emperror",
                self.detections,
                self.maps,
                {**self.generated, "anchor_bbox": self.detections.double()},
            )


if __name__ == "__main__":
    unittest.main()
