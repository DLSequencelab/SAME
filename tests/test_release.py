"""Unit and CPU integration tests for the released model and training protocol."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import train as runner
from same import SAME, SAMEConfig

torch.set_num_threads(1)


class ModelTests(unittest.TestCase):
    def config(self, **updates):
        return SAMEConfig(**{
            "channels": 3, "classes": 2, "patch_length": 8, "patch_stride": 4,
            "fft_bands": 2, "width": 8, "evidence_dim": 8,
            "expert_dim": 4, "fft_width": 4, "dropout": 0.0, **updates})

    def test_additive_reconstruction_and_weight_normalization(self):
        model = SAME(self.config()).eval()
        x = torch.randn(3, 23, 3)
        mask = torch.ones(3, 23, dtype=torch.bool)
        mask[1, 13:] = False
        with torch.no_grad():
            logits, aux = model(x, mask, return_aux=True)
        torch.testing.assert_close(logits, aux["signed_contributions"].sum((1, 2)), rtol=0, atol=0)
        available = aux["branch_available"]
        valid = available.any(-1)
        torch.testing.assert_close(aux["branch_weights"].sum(-1), valid.float())
        torch.testing.assert_close(aux["temporal_weights"].sum(1), available.any(1).float())
        self.assertTrue((aux["branch_weights"][~available] == 0).all())
        self.assertTrue((aux["temporal_weights"][~available] == 0).all())

    def test_masked_values_and_right_padding_do_not_change_prediction(self):
        model = SAME(self.config()).eval()
        x = torch.randn(2, 19, 3)
        mask = torch.ones(2, 19, dtype=torch.bool)
        mask[1, 12:] = False
        changed = x.clone()
        changed[~mask] = float("nan")
        with torch.no_grad():
            baseline = model(x, mask)
            torch.testing.assert_close(baseline, model(changed, mask), rtol=0, atol=0)
            padded = torch.nn.functional.pad(changed, (0, 0, 0, 12), value=float("nan"))
            padded_mask = torch.nn.functional.pad(mask, (0, 12), value=False)
            torch.testing.assert_close(baseline, model(padded, padded_mask), rtol=1e-5, atol=1e-6)

    def test_cosine_factorized_backward_and_auxiliary_schedule(self):
        model = SAME(self.config(channels=20, classifier_mode="cosine"))
        self.assertEqual(model.effective_patch_mode, "factorized")
        x = torch.randn(4, 19, 20)
        y = torch.tensor([0, 1, 0, 1])
        model.set_training_progress(0)
        logits, aux = model(x, return_aux=True)
        losses = model.loss(logits, y, aux)
        self.assertAlmostEqual(losses["branch_weight"].item(), model.config.branch_loss_start, places=6)
        losses["total"].backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        model.set_training_progress(1)
        self.assertAlmostEqual(model.loss(logits, y, aux)["branch_weight"].item(),
                               model.config.branch_loss_end, places=6)

    def test_short_sample_unavailable_spectrum_is_finite(self):
        model = SAME(self.config()).eval()
        x = torch.randn(2, 3, 3)
        mask = torch.tensor([[True, False, False], [True, True, True]])
        logits, aux = model(x, mask, return_aux=True)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertFalse(aux["branch_available"][0, :, 3].any())
        self.assertEqual(aux["branch_logits"][0, 3].abs().sum().item(), 0)

    def test_invalid_config_and_input_are_rejected(self):
        for changes in ({"patch_stride": 9}, {"fft_bands": 5}, {"dropout": 1},
                        {"long_dilations": []}, {"router_temperature": float("nan")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.config(**changes)
        with self.assertRaises(ValueError):
            SAME(self.config())(torch.ones(2, 8, 3), torch.zeros(2, 8, dtype=torch.bool))

    def test_all_dataset_model_recipes(self):
        recipes = runner.read_json(Path(__file__).resolve().parents[1] / "configs/uea18.json")
        self.assertEqual(len(recipes["datasets"]), 18)
        for dataset in recipes["datasets"]:
            model_recipe, training = runner.resolve_recipe(recipes, dataset, 42, SimpleNamespace())
            config = SAMEConfig(**model_recipe, channels=3, classes=2)
            self.assertEqual(training["protocol"], runner.PROTOCOL)
            self.assertEqual(training["epochs"], 100)
            self.assertEqual(training["patience"], 20)
            self.assertGreater(sum(p.numel() for p in SAME(config).parameters()), 0)


    def test_lamost_model_recipe(self):
        recipes = runner.read_json(Path(__file__).resolve().parents[1] / "configs/lamost.json")
        self.assertEqual(list(recipes["datasets"]), ["LAMOST"])
        model_recipe, training = runner.resolve_recipe(recipes, "LAMOST", 42, SimpleNamespace())
        config = SAMEConfig(**model_recipe, channels=1, classes=8)
        self.assertEqual((config.patch_length, config.patch_stride), (128, 16))
        self.assertFalse(config.normalize_input)
        self.assertEqual(training["protocol"], runner.PROTOCOL)
        self.assertEqual((training["epochs"], training["patience"]), (100, 100))
        model = SAME(config).eval()
        with torch.no_grad():
            logits, aux = model(torch.randn(2, 257, 1), return_aux=True)
        self.assertEqual(tuple(logits.shape), (2, 8))
        self.assertTrue(torch.isfinite(logits).all())
        torch.testing.assert_close(logits, aux["signed_contributions"].sum((1, 2)), rtol=0, atol=0)


class TrainingTests(unittest.TestCase):
    def make_data(self, folder):
        rng = np.random.default_rng(7)
        path = Path(folder) / "Toy.npz"
        x = rng.normal(size=(12, 12, 3)).astype(np.float32)
        mask = np.ones((12, 12), dtype=bool)
        mask[:, -1] = False
        x[:, -1] = np.nan
        np.savez(path, train_x=x, train_y=np.array(["A", "B"] * 6), train_mask=mask,
                 test_x=rng.normal(size=(6, 16, 3)).astype(np.float32),
                 test_y=np.array(["B", "A"] * 3))
        return path

    def test_original_split_and_train_label_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            arrays = runner.load_arrays(self.make_data(temporary))
            self.assertEqual(arrays[0].shape, (12, 12, 3))
            self.assertEqual(arrays[3].shape, (6, 16, 3))
            np.testing.assert_array_equal(arrays[1], [0, 1] * 6)
            np.testing.assert_array_equal(arrays[4], [1, 0] * 3)
            self.assertTrue((arrays[0][:, -1] == 0).all())

    def test_early_stopping_patience_reset_and_delta(self):
        stopper = runner.TrainingCEEarlyStopper(2, min_delta=.01)
        result = [stopper.update(v, e) for e, v in enumerate([1., .995, .98, .98, 1.], 1)]
        self.assertEqual(result, [(True, False), (False, False), (True, False),
                                  (False, False), (False, True)])
        self.assertEqual(stopper.reference_epoch, 3)

    def test_metrics_include_all_training_classes(self):
        metrics, confusion = runner.classification_metrics(
            np.array([[3., 0., 0.], [0., 3., 0.]]), np.array([0, 1]), 3)
        self.assertEqual(metrics["accuracy"], 1)
        self.assertAlmostEqual(metrics["macro_f1"], 2 / 3)
        self.assertEqual(confusion.shape, (3, 3))

    def run_checked_training(self, plateau):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.make_data(root)
            folder = root / "run"
            recipe = {"architecture_version": "ordered_multiscale_v2", "patch_length": 8,
                      "patch_stride": 4, "width": 8, "evidence_dim": 8, "expert_dim": 4,
                      "fft_width": 4, "fft_bands": 2}
            training = dict(runner.TRAIN_DEFAULTS, epochs=6 if plateau else 2,
                            patience=2 if plateau else 20, batch_size=4)
            expected_epoch = 3 if plateau else 2
            actual_evaluate = runner.evaluate
            original_update = runner.TrainingCEEarlyStopper.update
            def checked_evaluate(model, loader, device, count):
                self.assertTrue((folder / "final.pt").is_file())
                checkpoint = torch.load(folder / "final.pt", map_location="cpu", weights_only=False)
                self.assertEqual(checkpoint["epoch"], expected_epoch)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value.cpu(), checkpoint["state_dict"][name], rtol=0, atol=0)
                self.assertEqual(len(runner.read_json(folder / "history.json")), expected_epoch)
                self.assertEqual(len(loader.dataset), 6)
                return actual_evaluate(model, loader, device, count)
            def monitor(stopper, value, epoch):
                return original_update(stopper, 1.0 if plateau else value, epoch)
            with patch.object(runner, "evaluate", side_effect=checked_evaluate) as evaluation, \
                    patch.object(runner.TrainingCEEarlyStopper, "update", new=monitor), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = runner.train_one("Toy", 42, path, folder, recipe, training,
                                          "cpu", runner.sha256_file(path), "fixture")
            self.assertEqual(evaluation.call_count, 1)
            self.assertEqual(result["final_epoch"], expected_epoch)
            self.assertEqual(result["test_evaluations"], 1)
            self.assertFalse(result["test_used_for_selection"])
            self.assertEqual(result["stop_reason"], "train_ce_patience" if plateau else "max_epochs")
            self.assertFalse((folder / "best.pt").exists())
            history = runner.read_json(folder / "history.json")
            self.assertTrue(all(not any("test" in k or "validation" in k for k in row) for row in history))
            self.assertTrue(all("train_accuracy" in row and "learning_rate" in row for row in history))
            with np.load(folder / "test_outputs.npz") as saved:
                np.testing.assert_array_equal(saved["sample_indices"], np.arange(6))
                stored_logits = saved["logits"].copy()
            # The checkpoint is independently usable without the legacy project.
            checkpoint = torch.load(folder / "final.pt", map_location="cpu", weights_only=False)
            saved_model = SAME(SAMEConfig(**checkpoint["config"])).eval()
            saved_model.load_state_dict(checkpoint["state_dict"], strict=True)
            _, _, _, test_x, _, test_mask, _ = runner.load_arrays(path)
            with torch.no_grad():
                replay = saved_model(torch.from_numpy(test_x), torch.from_numpy(test_mask)).numpy()
            np.testing.assert_allclose(replay, stored_logits, rtol=1e-5, atol=1e-6)

    def test_max_epoch_budget_and_one_final_test(self):
        self.run_checked_training(plateau=False)

    def test_controlled_early_stop_retains_stopping_epoch(self):
        self.run_checked_training(plateau=True)

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileExistsError):
                runner.main(["--config", str(Path(__file__).resolve().parents[1] / "configs/uea18.json"),
                             "--output", str(root)])

    def test_cli_preflight_and_complete_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_data(root)
            config = root / "config.json"
            config.write_text(json.dumps({
                "defaults": {
                    "model": {
                        "architecture_version": "ordered_multiscale_v2",
                        "patch_length": 8, "patch_stride": 4, "width": 8,
                        "evidence_dim": 8, "expert_dim": 4, "fft_width": 4,
                        "fft_bands": 2,
                    },
                    "training": {"protocol": runner.PROTOCOL, "epochs": 2,
                                 "patience": 20, "batch_size": 4},
                },
                "datasets": {"Toy": {}},
            }), encoding="utf-8")
            output = root / "output"
            arguments = ["--config", str(config), "--data_root", str(root),
                         "--datasets", "Toy", "--seeds", "42", "--device", "cpu",
                         "--output", str(output)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(arguments + ["--dry_run"]), 0)
                self.assertFalse(output.exists())
                self.assertEqual(runner.main(arguments), 0)
            self.assertEqual(runner.read_json(output / "progress.json")["status"], "completed")
            summary = runner.read_json(output / "summary.json")
            self.assertEqual(summary["completed_runs"], 1)
            self.assertIsNone(summary["datasets"][0]["ACC_sample_sd_pct"])
            self.assertTrue((output / "seed_metrics.csv").is_file())
            self.assertTrue((output / "data_manifests/Toy.json").is_file())
            manifest = runner.read_json(output / "experiment_manifest.json")
            self.assertIn("same/model.py", manifest["model_sources_sha256"])
            self.assertFalse(manifest["validation_used"])


if __name__ == "__main__":
    unittest.main()
