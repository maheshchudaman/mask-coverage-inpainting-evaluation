#!/usr/bin/env python3
"""Fast mathematical and determinism checks that do not require ADE20K."""

import math
import random

import numpy as np

from run_study import controlled_completion, make_mask, metrics


def main():
    size = 64
    yy, xx = np.mgrid[0:size, 0:size]
    reference = np.stack(
        [xx / (size - 1), yy / (size - 1), (xx + yy) / (2 * (size - 1))], axis=2
    ).astype(np.float32)
    config = {
        "blur_radius": 3.0,
        "shift_fraction": 0.12,
        "noise_sigma": 0.20,
    }
    for family in ("center", "rectangle", "freeform"):
        for coverage in (0.10, 0.30, 0.50):
            first = make_mask(family, size, coverage, random.Random(123), 0.015)
            second = make_mask(family, size, coverage, random.Random(123), 0.015)
            assert np.array_equal(first, second), (family, coverage, "non-deterministic mask")
            assert abs(float(first.mean()) - coverage) <= 1 / first.size + 1e-6
            for operator in ("mean", "blur", "noise", "shift"):
                output = controlled_completion(
                    reference, first, operator, random.Random(456), config
                )
                result = metrics(reference, output, first)
                assert result["observed_max_error"] == 0.0
                # Exact identity when observed pixels are copied unchanged:
                # full MSE = realised hole fraction * hole MSE.
                expected = float(first.mean()) * result["hole_mse"]
                assert math.isclose(result["full_mse"], expected, rel_tol=2e-6, abs_tol=1e-10)
                expected_gap = -10.0 * math.log10(float(first.mean()))
                actual_gap = result["full_psnr"] - result["hole_psnr"]
                assert math.isclose(actual_gap, expected_gap, rel_tol=2e-6, abs_tol=2e-6)
    print("self-check passed")


if __name__ == "__main__":
    main()
