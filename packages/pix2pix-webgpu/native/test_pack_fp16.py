"""Regression tests for the packed model's deterministic byte layout."""
import unittest
from array import array
from pack_fp16 import pack_values, layout, PACKED_BYTES, TOTAL_FLOATS


class PackedWeightTests(unittest.TestCase):
    def test_layout_preserves_bias_and_normalization_as_fp32(self):
        spans = layout()
        self.assertEqual(sum(count for _, count, _ in spans), TOTAL_FLOATS)
        self.assertEqual(sum(count * (2 if is_conv else 4)
                             for _, count, is_conv in spans), PACKED_BYTES)
        self.assertEqual(spans[0], ("e1_weights", 3072, True))
        self.assertIn(("e1_bias", 64, False), spans)

    def test_fp16_only_for_convolution_span(self):
        payload = pack_values(array("f", [1.0, -0.33333334, 2.0]), True)
        self.assertEqual(len(payload), 6)
        self.assertEqual(payload[:2], bytes.fromhex("003c"))
        self.assertEqual(pack_values(array("f", [1.0, -0.33333334]), False)[:4],
                         bytes.fromhex("0000803f"))


if __name__ == "__main__":
    unittest.main()
