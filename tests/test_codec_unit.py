"""Model-free codec unit tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from llz import cli, codec_lib
from llz.errors import ConfigurationError


class QuantizerTests(unittest.TestCase):
    """Verify quantizer invariants and invalid-input handling."""

    def setUp(self) -> None:
        """Save the mutable process-wide scale."""
        self.original_scale = codec_lib.SCALE

    def tearDown(self) -> None:
        """Restore the mutable process-wide scale."""
        codec_lib.SCALE = self.original_scale

    def test_quantizer_produces_positive_monotonic_counts(self) -> None:
        """Every symbol receives at least one count and total stays bounded."""
        codec_lib.SCALE = 1 << 16
        probabilities = np.array([0.0, 0.1, 0.2, 0.7])

        cumulative, total = codec_lib.quantize(probabilities, 4)

        frequencies = np.diff(np.concatenate(([0], cumulative)))
        self.assertTrue((frequencies >= 1).all())
        self.assertTrue((np.diff(cumulative) > 0).all())
        self.assertLessEqual(total, codec_lib.SCALE)

    def test_quantizer_rejects_non_finite_values(self) -> None:
        """NaN and infinity cannot enter arithmetic coding."""
        with self.assertRaises(ConfigurationError):
            codec_lib.quantize(np.array([0.5, np.nan]), 2)
        with self.assertRaises(ConfigurationError):
            codec_lib.quantize(np.array([0.5, np.inf]), 2)

    def test_quantizer_rejects_invalid_shape_and_negative_values(self) -> None:
        """Malformed probability vectors fail explicitly."""
        with self.assertRaises(ConfigurationError):
            codec_lib.quantize(np.array([1.0]), 2)
        with self.assertRaises(ConfigurationError):
            codec_lib.quantize(np.array([1.1, -0.1]), 2)

    def test_scale_validation_survives_optimized_python(self) -> None:
        """Invalid scales use exceptions, not removable assertions."""
        with self.assertRaises(ConfigurationError):
            cli._validate_scale(15, 50_257, 30)
        with self.assertRaises(ConfigurationError):
            cli._validate_scale(29, 50_257, 30)


class ModelCacheTests(unittest.TestCase):
    """Verify models are cached by id and immutable revision."""

    def setUp(self) -> None:
        """Clear process caches before each test."""
        cli._state.clear()

    @patch("llz.codec_lib.resolve_model_revision", return_value=None)
    @patch("llz.codec_lib.get_model")
    def test_distinct_models_do_not_share_cached_weights(
        self, get_model, _resolve_revision
    ) -> None:
        """Switching model ids in one process loads the requested model."""
        get_model.side_effect = [("tok-a", "model-a"), ("tok-b", "model-b")]

        first = cli.get_codec("org/a", "rev-a")
        second = cli.get_codec("org/b", "rev-b")
        again = cli.get_codec("org/a", "rev-a")

        self.assertEqual(first[:2], ("tok-a", "model-a"))
        self.assertEqual(second[:2], ("tok-b", "model-b"))
        self.assertEqual(again[:2], ("tok-a", "model-a"))
        self.assertEqual(get_model.call_count, 2)

    @patch("llz.codec_lib.resolve_model_revision", return_value="commit-a")
    @patch("llz.codec_lib.get_model", return_value=("tok-a", "model-a"))
    def test_resolved_revision_reuses_loaded_model(
        self, get_model, _resolve_revision
    ) -> None:
        """A container's resolved revision does not load duplicate weights."""
        first = cli.get_codec("org/a")
        exact = cli.get_codec("org/a", "commit-a")

        self.assertIs(first, exact)
        get_model.assert_called_once_with("org/a", None)


if __name__ == "__main__":
    unittest.main()
