import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts import lookalign_mvp as la


def save_image(path: Path, img: np.ndarray) -> None:
    Image.fromarray((np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="RGB").save(path)


def synthetic_source(size: int = 96) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    checker = ((x // 12 + y // 12) % 2).astype(np.float32)
    ramp = x / max(size - 1, 1)
    lum = 0.10 + 0.78 * (0.72 * checker + 0.28 * ramp)
    return np.stack([lum * 1.04, lum * 0.96, lum * 0.90], axis=-1).astype(np.float32)


def low_contrast_reference(size: int = 96) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    soft = 0.50 + 0.08 * np.sin(x / 9.0) + 0.05 * np.cos(y / 11.0)
    return np.stack([soft * 0.90, soft * 1.06, soft * 1.22], axis=-1).astype(np.float32)


def mean_lab_ab(img: np.ndarray) -> np.ndarray:
    lab, _ = la.rgb_to_lab(img.astype(np.float32))
    return lab[..., 1:3].reshape(-1, 2).mean(axis=0)


def warm_reference(size: int = 96) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    base = 0.52 + 0.10 * np.sin(x / 8.5) + 0.07 * np.cos(y / 10.5)
    return np.stack([base * 1.18, base * 0.98, base * 0.86], axis=-1).astype(np.float32)


class LookAlignAntiFadeTests(unittest.TestCase):
    def run_synthetic(self, source: np.ndarray, reference: np.ndarray, **config: object) -> np.ndarray:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src_path = tmp_path / "source.png"
            ref_path = tmp_path / "reference.png"
            out_path = tmp_path / "output.png"
            save_image(src_path, source)
            save_image(ref_path, reference)
            run_config = {
                "align": "identity",
                "blur_sigma": 5.0,
                "grid": 6,
                "trust_threshold": 0.0,
                "debug_dir": None,
            }
            run_config.update(config)
            la.run_lookalign(src_path, ref_path, out_path, run_config)
            return la.load_rgb(str(out_path))

    def test_high_contrast_source_does_not_become_low_contrast(self) -> None:
        source = synthetic_source()
        reference = low_contrast_reference()
        output = self.run_synthetic(source, reference)
        source_std = la.luminance(source).std()
        output_std = la.luminance(output).std()
        self.assertGreaterEqual(output_std, source_std * 0.83)

    def test_default_luminance_std_floor_is_enforced(self) -> None:
        source = synthetic_source()
        reference = np.full_like(source, [0.42, 0.44, 0.48], dtype=np.float32)
        output = self.run_synthetic(source, reference)
        ratio = la.luminance(output).std() / max(float(la.luminance(source).std()), la.EPS)
        self.assertGreaterEqual(ratio, 0.84)

    def test_chroma_transfer_moves_toward_reference(self) -> None:
        source = synthetic_source()
        reference = low_contrast_reference()
        output = self.run_synthetic(source, reference)
        src_dist = np.linalg.norm(mean_lab_ab(source) - mean_lab_ab(reference))
        out_dist = np.linalg.norm(mean_lab_ab(output) - mean_lab_ab(reference))
        self.assertLess(out_dist, src_dist * 0.85)

    def test_zero_local_luma_strength_preserves_local_luminance_structure(self) -> None:
        size = 80
        source = synthetic_source(size)
        y, x = np.mgrid[0:size, 0:size].astype(np.float32)
        ref = source.copy()
        ref[..., 0] = np.clip(ref[..., 0] + 0.16 * (x > size / 2), 0.0, 1.0)
        ref[..., 2] = np.clip(ref[..., 2] + 0.18 * (y > size / 2), 0.0, 1.0)
        mask = np.ones((size, size), dtype=np.float32)
        weights = np.ones((size, size), dtype=np.float32)

        _, local_result, enabled = la.compute_local_field(source, ref, mask, weights, grid=4, blur_sigma=4.0, local_luma_strength=0.0, warnings=[])

        source_lab, _ = la.rgb_to_lab(source)
        local_lab, _ = la.rgb_to_lab(local_result)
        l_delta = np.abs(local_lab[..., 0] - source_lab[..., 0]).mean()
        ab_delta = np.abs(local_lab[..., 1:3] - source_lab[..., 1:3]).mean()
        self.assertTrue(enabled)
        self.assertLess(l_delta, 0.012)
        self.assertGreater(ab_delta, 0.004)

    def test_neutral_gray_source_does_not_turn_red(self) -> None:
        source = np.full((96, 96, 3), 0.52, dtype=np.float32)
        reference = warm_reference(96)
        output = self.run_synthetic(source, reference)
        out_ab = mean_lab_ab(output)
        self.assertLess(abs(float(out_ab[0]) - 0.5), 0.02)
        self.assertLess(abs(float(out_ab[1]) - 0.5), 0.025)

    def test_white_and_gray_patches_remain_near_neutral(self) -> None:
        source = np.full((96, 96, 3), 0.50, dtype=np.float32)
        source[:, :32] = 0.96
        source[:, 32:64] = 0.70
        source[:, 64:] = 0.18
        reference = warm_reference(96)
        output = self.run_synthetic(source, reference)
        out_lab, _ = la.rgb_to_lab(output)
        for sl in [slice(0, 32), slice(32, 64), slice(64, 96)]:
            patch_ab = out_lab[:, sl, 1:3].reshape(-1, 2).mean(axis=0)
            self.assertLess(abs(float(patch_ab[0]) - 0.5), 0.025)
            self.assertLess(abs(float(patch_ab[1]) - 0.5), 0.03)

    def test_chroma_moves_in_non_neutral_regions(self) -> None:
        size = 96
        source = np.full((size, size, 3), [0.22, 0.48, 0.24], dtype=np.float32)
        source[:, size // 2 :] = [0.26, 0.38, 0.70]
        reference = np.full((size, size, 3), [0.62, 0.38, 0.20], dtype=np.float32)
        reference[:, size // 2 :] = [0.72, 0.52, 0.28]
        output = self.run_synthetic(source, reference)
        src_dist = np.linalg.norm(mean_lab_ab(source) - mean_lab_ab(reference))
        out_dist = np.linalg.norm(mean_lab_ab(output) - mean_lab_ab(reference))
        self.assertLess(out_dist, src_dist * 0.90)


if __name__ == "__main__":
    unittest.main()
