import unittest

import torch

from costeer.optimizer import CoSteerConfig, CoSteerOptimizer


class CoSteerOptimizerTest(unittest.TestCase):
    def test_zero_delta_preserves_base_policy(self) -> None:
        llm = torch.tensor([[1.0, 0.0, -1.0]])
        slm = torch.tensor([[0.1, 0.2, 0.3]])
        optimizer = CoSteerOptimizer(CoSteerConfig(iterations=5))

        actual = optimizer.optimize_policy(llm, slm, slm)
        expected = torch.log_softmax(llm, dim=-1)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_output_is_a_normalized_log_policy(self) -> None:
        optimizer = CoSteerOptimizer()
        actual = optimizer.optimize_policy(
            torch.tensor([[0.5, 0.2, -0.1]]),
            torch.tensor([[0.4, 0.3, 0.1]]),
            torch.tensor([[0.1, 0.3, 0.8]]),
        )

        self.assertEqual(tuple(actual.shape), (1, 3))
        self.assertTrue(
            torch.allclose(torch.logsumexp(actual, dim=-1), torch.zeros(1), atol=1e-6)
        )

    def test_mismatched_vocabularies_are_rejected(self) -> None:
        optimizer = CoSteerOptimizer()
        with self.assertRaisesRegex(ValueError, "same vocabulary"):
            optimizer.optimize_policy(
                torch.zeros(1, 4), torch.zeros(1, 3), torch.zeros(1, 3)
            )


if __name__ == "__main__":
    unittest.main()
