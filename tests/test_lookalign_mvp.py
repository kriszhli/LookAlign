import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
from PIL import Image

import app
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
                "base_radius": 5.0,
                "light_map_grid": 6,
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

    def test_light_map_only_changes_luminance(self) -> None:
        size = 80
        source = synthetic_source(size)
        y, x = np.mgrid[0:size, 0:size].astype(np.float32)
        ref = source.copy()
        ref[..., 0] = np.clip(ref[..., 0] + 0.16 * (x > size / 2), 0.0, 1.0)
        ref[..., 2] = np.clip(ref[..., 2] + 0.18 * (y > size / 2), 0.0, 1.0)
        mask = np.ones((size, size), dtype=np.float32)
        weights = np.ones((size, size), dtype=np.float32)

        maps, stats = la.compute_light_map(
            source,
            source,
            ref,
            mask,
            weights,
            light_map_grid=4,
            light_map_smooth=0.35,
            light_map_radius=3.0,
            max_exposure_delta=0.18,
            max_contrast_gain=1.35,
            guide_eps=0.01,
            upsample_mode="source_guided",
            warnings=[],
        )
        local_result, _ = la.apply_light_map_torch(source, mask, maps, local_strength=1.0, local_luma_strength=1.0, device_name="cpu")

        source_lab, _ = la.rgb_to_lab(source)
        local_lab, _ = la.rgb_to_lab(local_result)
        l_delta = np.abs(local_lab[..., 0] - source_lab[..., 0]).mean()
        ab_delta = np.abs(local_lab[..., 1:3] - source_lab[..., 1:3]).mean()
        self.assertTrue(stats["enabled"])
        self.assertGreater(l_delta, 0.004)
        self.assertLess(ab_delta, 0.003)

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


class LookAlignSALUTTests(unittest.TestCase):
    def test_sa_lut_global_match_reduces_spatial_low_frequency_error(self) -> None:
        size = 64
        y, x = np.mgrid[0:size, 0:size].astype(np.float32)
        source = np.stack(
            [
                0.20 + 0.55 * x / max(size - 1, 1),
                0.30 + 0.35 * y / max(size - 1, 1),
                0.32 + 0.22 * np.sin(x / 9.0),
            ],
            axis=-1,
        ).astype(np.float32)
        source = np.clip(source, 0.0, 1.0)
        reference = source.copy()
        reference[:, : size // 2, 0] = np.clip(reference[:, : size // 2, 0] + 0.16, 0.0, 1.0)
        reference[:, : size // 2, 2] = np.clip(reference[:, : size // 2, 2] - 0.07, 0.0, 1.0)
        reference[:, size // 2 :, 0] = np.clip(reference[:, size // 2 :, 0] - 0.08, 0.0, 1.0)
        reference[:, size // 2 :, 2] = np.clip(reference[:, size // 2 :, 2] + 0.14, 0.0, 1.0)
        weights = np.ones((size, size), dtype=np.float32)
        warnings: list[str] = []
        args = la._config_to_namespace(
            {
                "sa_lut_size": 9,
                "sa_lut_context_bins": 2,
                "sa_lut_fit_max_samples": 0,
                "sa_lut_ridge": 0.01,
                "sa_lut_smooth": 0.25,
                "render_backend": "pytorch",
            }
        )

        matched, params, context, base_matched = la.sa_lut_global_color_transfer(source, source, reference, weights, weights, args, warnings)

        self.assertEqual(params["method"], "sa_lut")
        self.assertEqual(context.shape, weights.shape)
        self.assertEqual(base_matched.shape, source.shape)
        before = float(np.mean((source - reference) ** 2))
        after = float(np.mean((matched - reference) ** 2))
        self.assertLess(after, before * 0.75)

    def test_sa_lut_insufficient_overlap_falls_back_without_crashing(self) -> None:
        source = synthetic_source(32)
        reference = warm_reference(32)
        weights = np.zeros((32, 32), dtype=np.float32)
        weights[0:2, 0:2] = 1.0
        warnings: list[str] = []
        args = la._config_to_namespace({"render_backend": "pytorch"})

        matched, params, context, base_matched = la.sa_lut_global_color_transfer(source, source, reference, np.ones((32, 32), dtype=np.float32), weights, args, warnings)

        self.assertEqual(matched.shape, source.shape)
        self.assertEqual(context.shape, weights.shape)
        self.assertEqual(base_matched.shape, source.shape)
        self.assertEqual(params["method"], "sa_lut")
        self.assertFalse(params["enabled"])
        self.assertLess(float(np.mean(np.abs(matched - source))), 0.001)
        self.assertTrue(any("SA-LUT fitting skipped" in msg for msg in warnings))

    def test_light_map_reduces_local_luminance_error(self) -> None:
        size = 72
        source = synthetic_source(size)
        y, _ = np.mgrid[0:size, 0:size].astype(np.float32)
        reference = np.clip(source * (0.80 + 0.35 * y[..., None] / max(size - 1, 1)), 0.0, 1.0)
        mask = np.ones((size, size), dtype=np.float32)
        weights = np.ones((size, size), dtype=np.float32)
        maps, stats = la.compute_light_map(
            source,
            source,
            reference,
            mask,
            weights,
            light_map_grid=6,
            light_map_smooth=0.35,
            light_map_radius=4.0,
            max_exposure_delta=0.18,
            max_contrast_gain=1.35,
            guide_eps=0.01,
            upsample_mode="source_guided",
            warnings=[],
        )
        adjusted, _ = la.apply_light_map_torch(source, mask, maps, local_strength=1.0, local_luma_strength=1.0, device_name="cpu")
        before = float(np.mean((la.luminance(source) - la.luminance(reference)) ** 2))
        after = float(np.mean((la.luminance(adjusted) - la.luminance(reference)) ** 2))
        self.assertTrue(stats["enabled"])
        self.assertLess(after, before * 0.65)

    def test_light_map_grid_stays_compute_bounded(self) -> None:
        size = 84
        source = synthetic_source(size)
        reference = np.clip(source + 0.08, 0.0, 1.0)
        mask = np.ones((size, size), dtype=np.float32)
        weights = np.ones((size, size), dtype=np.float32)

        maps, stats = la.compute_light_map(
            source,
            source,
            reference,
            mask,
            weights,
            light_map_grid=7,
            light_map_smooth=0.0,
            light_map_radius=3.0,
            max_exposure_delta=0.18,
            max_contrast_gain=1.35,
            guide_eps=0.01,
            upsample_mode="source_guided",
            warnings=[],
        )

        self.assertTrue(stats["enabled"])
        self.assertEqual(maps["coarse_light_map"].shape[0], 7)
        self.assertLessEqual(maps["coarse_light_map"].size, 7 * 8)
        self.assertEqual(maps["light_map"].shape, mask.shape)

    def test_source_guided_light_map_respects_luminance_edge(self) -> None:
        size = 80
        source = np.full((size, size, 3), 0.28, dtype=np.float32)
        source[:, size // 2 :] = 0.72
        reference = source.copy()
        reference[:, : size // 2] = np.clip(reference[:, : size // 2] + 0.14, 0.0, 1.0)
        mask = np.ones((size, size), dtype=np.float32)
        weights = np.ones((size, size), dtype=np.float32)

        maps, stats = la.compute_light_map(
            source,
            source,
            reference,
            mask,
            weights,
            light_map_grid=5,
            light_map_smooth=0.0,
            light_map_radius=6.0,
            max_exposure_delta=0.18,
            max_contrast_gain=1.35,
            guide_eps=0.002,
            upsample_mode="source_guided",
            warnings=[],
        )

        left = float(maps["light_map"][:, : size // 2 - 4].mean())
        right = float(maps["light_map"][:, size // 2 + 4 :].mean())
        self.assertTrue(stats["enabled"])
        self.assertGreater(left, right + 0.04)

    def test_gradio_run_ui_returns_v025_debug_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src_path = tmp_path / "source.png"
            ref_path = tmp_path / "reference.png"
            save_image(src_path, synthetic_source(48))
            save_image(ref_path, warm_reference(48))
            with mock.patch.object(app, "OUTPUTS_DIR", tmp_path / "outputs"):
                result = app.run_ui(
                    str(src_path),
                    str(ref_path),
                    "identity",
                    0.4,
                    0.25,
                    1.0,
                    6.0,
                    4,
                    0.0,
                    0.85,
                    0.80,
                    1.0,
                    "SA-LUT Base",
                    "LookAlign Output",
                )
            paths = result[-1]
            self.assertIn("SA-LUT Base", paths)
            self.assertIn("Light Map Result", paths)
            self.assertTrue(Path(paths["SA-LUT Base"]).exists())

    @unittest.skipIf(la.torch is None, "PyTorch unavailable")
    def test_sa_lut_device_selector_prefers_mps_when_available(self) -> None:
        with mock.patch.object(la.torch.backends.mps, "is_built", return_value=True), mock.patch.object(la.torch.backends.mps, "is_available", return_value=True):
            device, info = la.choose_sa_lut_device(prefer_mps=True)
        self.assertEqual(device, "mps")
        self.assertTrue(info["mps_available"])

    @unittest.skipIf(la.torch is None, "PyTorch unavailable")
    def test_sa_lut_device_selector_falls_back_to_cpu(self) -> None:
        with mock.patch.object(la.torch.backends.mps, "is_built", return_value=True), mock.patch.object(la.torch.backends.mps, "is_available", return_value=False):
            device, info = la.choose_sa_lut_device(prefer_mps=True)
        self.assertEqual(device, "cpu")
        self.assertFalse(info["mps_available"])


if __name__ == "__main__":
    unittest.main()
