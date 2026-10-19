import tempfile
from pathlib import Path
import unittest

import numpy as np
from scipy.optimize import check_grad
import torch

from speech_ci.audio import add_noise
from speech_ci.descriptors import describe, layer_mean, shuffle_bins, valid_pairs
from speech_ci.events import extract as event_features, shuffle as shuffle_events
from speech_ci.event_heads import BlockHeadPath
from speech_ci.fusion import binary_fusion, from_margins, probability_margins
from speech_ci.heads import PairedHead
from speech_ci.metrics import classification_metrics, normalized_logit_drift
from speech_ci.models import BinaryReadout, MulticlassReadout, class_weights


def sequence():
    rng = np.random.default_rng(11)
    return {
        "features": rng.normal(size=(7, 12, 768)).astype(np.float32),
        "starts": np.arange(7, dtype=float) * 2,
        "ends": np.r_[np.arange(1, 7, dtype=float) * 2, 12.5],
        "token_counts": np.array([99, 100, 100, 100, 100, 100, 0]),
    }


class DescriptorTests(unittest.TestCase):
    def test_context_boundaries_and_fallback_bins_are_excluded(self):
        np.testing.assert_array_equal(valid_pairs(sequence()), [0, 1, 2, 3])

    def test_weighted_rms_definition(self):
        seq = sequence()
        values = layer_mean(seq, "wavlm")
        expected = np.sqrt(np.mean(np.diff(values[:5], axis=0) ** 2, axis=0))
        np.testing.assert_array_equal(describe(seq, "wavlm")["delta"], expected)

    def test_shuffle_preserves_partial_bins_and_marginal_statistics(self):
        seq = sequence()
        original = describe(seq, "wavlm")
        shuffled = shuffle_bins(seq, "synthetic", "CTD", 29)
        np.testing.assert_array_equal(shuffled["features"][-1], seq["features"][-1])
        for key in ("mean", "std"):
            np.testing.assert_allclose(describe(shuffled, "wavlm")[key], original[key], atol=1e-12)
        np.testing.assert_array_equal(
            shuffled["features"], shuffle_bins(seq, "synthetic", "CTD", 29)["features"]
        )

    def test_noise_has_fixed_snr_and_independent_views(self):
        audio = np.sin(np.arange(16000) * 0.013).astype(np.float32) * 0.2
        train = add_noise(audio, "synthetic", "CTD", "fit20")
        evaluation = add_noise(audio, "synthetic", "CTD", "eval20")
        np.testing.assert_array_equal(train, add_noise(audio, "synthetic", "CTD", "fit20"))
        self.assertFalse(np.array_equal(train, evaluation))
        snr = 20 * np.log10(np.sqrt(np.mean(audio ** 2)) / np.sqrt(np.mean((train - audio) ** 2)))
        self.assertAlmostEqual(float(snr), 20.0, places=4)


class HeadTests(unittest.TestCase):
    def test_binary_gradient(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(12, 5))
        noisy = x + rng.normal(size=x.shape) * 0.1
        y = np.arange(12) % 2
        weights = class_weights(y, 2)[y]
        head = PairedHead(c=0.1)
        p = rng.normal(size=6) * 0.1
        error = check_grad(
            lambda v: head.objective(v, x, noisy, y, weights)[0],
            lambda v: head.objective(v, x, noisy, y, weights)[1], p,
        )
        self.assertLess(error, 1e-5)

    def test_multiclass_gradient(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(12, 5))
        noisy = x + rng.normal(size=x.shape) * 0.1
        y = np.arange(12) % 3
        weights = class_weights(y, 3)[y]
        head = PairedHead(c=0.1, classes=3)
        p = rng.normal(size=18) * 0.1
        error = check_grad(
            lambda v: head.objective(v, x, noisy, y, weights)[0],
            lambda v: head.objective(v, x, noisy, y, weights)[1], p,
        )
        self.assertLess(error, 1e-5)

    def test_head_serialization_is_prediction_preserving(self):
        rng = np.random.default_rng(9)
        original = rng.normal(size=(18, 7))
        noisy = original + rng.normal(size=original.shape) * 0.1
        for classes in (2, 3):
            head = PairedHead(classes=classes).fit(original, noisy, np.arange(18) % classes)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "head.npz"
                head.save(path)
                restored = PairedHead.load(path)
                np.testing.assert_array_equal(head.predict_proba(original), restored.predict_proba(original))

    def test_block_normalization_preserves_equivalent_fixed_regularization(self):
        rng = np.random.default_rng(19)
        original = rng.normal(size=(18, 7))
        noisy = original + rng.normal(size=original.shape) * 0.1
        y = np.arange(18) % 2
        reference = PairedHead().fit(original, noisy, y)
        path = BlockHeadPath().fit({"x": original}, {"x": noisy}, y, [0.007])
        np.testing.assert_allclose(
            path.logits({"x": original}, 0.007), reference.decision_function(original), atol=2e-6, rtol=0
        )


class ReadoutTests(unittest.TestCase):
    def test_padding_does_not_change_predictions(self):
        torch.manual_seed(7)
        x = torch.randn(2, 3, 5, 768)
        duration = torch.full((2, 3, 5), 2.0)
        for model in (BinaryReadout(), MulticlassReadout()):
            model.eval()
            with torch.no_grad():
                original = model(x, duration)["logits"]
                padded_x = torch.nn.functional.pad(x, (0, 0, 0, 3))
                padded_duration = torch.nn.functional.pad(duration, (0, 3))
                padded = model(padded_x, padded_duration)["logits"]
            torch.testing.assert_close(original, padded, atol=1e-6, rtol=1e-6)

    def test_multiclass_margin_conversion(self):
        probability = np.array([[0.1, 0.3, 0.6], [0.5, 0.25, 0.25]])
        np.testing.assert_allclose(from_margins(probability_margins(probability)), probability, atol=1e-15)

    def test_fusion_zero_auxiliaries(self):
        probabilities = binary_fusion(np.full(4, 0.5), np.zeros(4), np.zeros(4))
        np.testing.assert_array_equal(probabilities["Full"], np.full(4, 0.5))

    def test_normalized_drift_is_affine_invariant(self):
        fitting = np.array([-2.0, -1.0, 0.0, 2.0])
        original = np.array([-0.3, 0.2])
        noisy = np.array([-0.2, 0.5])
        expected = normalized_logit_drift(fitting, original, noisy)
        actual = normalized_logit_drift(7 * fitting + 2, 7 * original + 2, 7 * noisy + 2)
        self.assertAlmostEqual(expected, actual, places=14)

    def test_binary_metric_class_order(self):
        result = classification_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.3, 0.8]))
        self.assertEqual(result["confusion_matrix"], [[2, 0], [1, 1]])
        self.assertEqual(result["hc_specificity"], 1.0)
        self.assertEqual(result["ci_recall"], 0.5)


class EventTests(unittest.TestCase):
    def test_event_shapes_and_empty_audio(self):
        for segments in ([], [[0.0, 12.0]], [[1.0, 2.0], [4.0, 7.0]]):
            summary, temporal = event_features(segments, 12.0)
            self.assertEqual(summary.shape, (21,))
            self.assertEqual(temporal.shape, (18,))
            self.assertTrue(np.isfinite(np.r_[summary, temporal]).all())

    def test_event_shuffle_preserves_summary_features(self):
        segments = np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 8.0], [9.0, 11.0]])
        original = event_features(segments, 12.0)[0]
        for seed in (11, 29, 47):
            shuffled = shuffle_events(segments, 12.0, "synthetic", "CTD", seed)
            np.testing.assert_allclose(event_features(shuffled, 12.0)[0], original, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
