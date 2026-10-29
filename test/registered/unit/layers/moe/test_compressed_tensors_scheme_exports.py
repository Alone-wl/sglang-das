"""CPU regressions for compressed-tensors scheme package exports."""

import unittest

from sglang.srt.layers.quantization.compressed_tensors import (
    compressed_tensors,
    schemes,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a8_int8_hcu_moe import (
    HCUCompressedTensorsW4A8Int8DynamicMoE,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a8_int8_moe import (
    NPUCompressedTensorsW4A8Int8DynamicMoE,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCompressedTensorsSchemeExports(CustomTestCase):
    def test_hcu_w4a8_export_uses_dedicated_scheme(self):
        self.assertIs(
            schemes.HCUCompressedTensorsW4A8Int8DynamicMoE,
            HCUCompressedTensorsW4A8Int8DynamicMoE,
        )
        self.assertIs(
            compressed_tensors.HCUCompressedTensorsW4A8Int8DynamicMoE,
            HCUCompressedTensorsW4A8Int8DynamicMoE,
        )

    def test_npu_w4a8_export_is_preserved(self):
        self.assertIs(
            schemes.NPUCompressedTensorsW4A8Int8DynamicMoE,
            NPUCompressedTensorsW4A8Int8DynamicMoE,
        )


if __name__ == "__main__":
    unittest.main()
