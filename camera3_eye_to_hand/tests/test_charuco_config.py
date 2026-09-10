from __future__ import annotations

import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import yaml

MODULE_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIRECTORY))

from calibrate_camera3 import (  # noqa: E402
    CalibrationSession,
    CaptureRecord,
    CharucoPoseDetector,
)


class CharucoConfigTests(unittest.TestCase):
    def test_config_detects_board_generated_from_its_settings(self) -> None:
        config = yaml.safe_load((MODULE_DIRECTORY / "config.yaml").read_text())
        detector = CharucoPoseDetector(config["charuco"])
        board_width = config["charuco"]["squares_x"] * 200
        board_height = config["charuco"]["squares_y"] * 200
        board_gray = detector.board.generateImage((board_width, board_height))
        board_rgb = cv2.cvtColor(board_gray, cv2.COLOR_GRAY2RGB)
        board_rgb = cv2.copyMakeBorder(
            board_rgb,
            300,
            300,
            300,
            300,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

        camera_matrix = np.array(
            [
                [1600.0, 0.0, (board_width + 600) / 2],
                [0.0, 1600.0, (board_height + 600) / 2],
                [0.0, 0.0, 1.0],
            ]
        )
        result = detector.detect(board_rgb, camera_matrix, np.zeros(5))

        expected_corners = (
            (config["charuco"]["squares_x"] - 1)
            * (config["charuco"]["squares_y"] - 1)
        )
        self.assertEqual(result.num_corners, expected_corners)
        self.assertIsNotNone(result.T_camera_board)
        self.assertLess(result.reprojection_error_px, 0.1)

    def test_result_files_can_be_written_without_hardware(self) -> None:
        config_path = MODULE_DIRECTORY / "config.yaml"
        config = deepcopy(yaml.safe_load(config_path.read_text()))
        identity = np.eye(4)

        with tempfile.TemporaryDirectory() as temporary_directory:
            config["calibration"]["output_dir"] = temporary_directory
            session = CalibrationSession(config, config_path)
            session.records = [
                CaptureRecord(
                    T_base_camera3=identity,
                    T_base_ee=identity,
                    T_camera2_board=identity,
                    T_camera3_board=identity,
                    camera2_reprojection_error_px=0.1,
                    camera3_reprojection_error_px=0.1,
                    valid_frames=10,
                )
                for _ in range(config["calibration"]["minimum_captures"])
            ]
            matrix_path, report_path, captures_path = session.save()

            self.assertTrue(matrix_path.is_file())
            self.assertTrue(report_path.is_file())
            self.assertTrue(captures_path.is_file())
            self.assertEqual(matrix_path.parent.name, "result_1")
            report = yaml.safe_load(report_path.read_text())
            self.assertEqual(report["transform_name"], "T_base_camera3")
            np.testing.assert_allclose(np.load(matrix_path), identity)

            second_matrix_path, _, _ = session.save()
            self.assertEqual(second_matrix_path.parent.name, "result_2")


if __name__ == "__main__":
    unittest.main()
