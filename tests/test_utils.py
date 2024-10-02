import unittest

from annotator_diversity.datasets import generate_soft_labels_raw_annotations
from preprocess_data import compute_normalized_entropy


class TestNormalizedEntropy(unittest.TestCase):
    def test_max_disagreement(self):
        # For 3 labels: [1, 2, 3], each label has equal probability leading to max disagreement
        annotations = [1, 2, 3]
        num_labels = 3
        expected_entropy = 1.0
        self.assertEqual(
            compute_normalized_entropy(annotations, num_labels), expected_entropy
        )

    def test_min_disagreement(self):
        # All annotators agree on one label, so no disagreement
        annotations = [1, 1, 1]
        num_labels = 3
        expected_entropy = 0.0
        self.assertEqual(
            compute_normalized_entropy(annotations, num_labels), expected_entropy
        )

    def test_intermediate_disagreement(self):
        # Some intermediate level of disagreement
        annotations = [1, 1, 2, 2, 3]
        num_labels = 3
        # The expected value is calculated manually or using some reference
        expected_entropy = (
            0.9602297178607612  # This is the actual entropy value for these annotations
        )
        self.assertAlmostEqual(
            compute_normalized_entropy(annotations, num_labels), expected_entropy
        )


class TestGenerateSoftLabels(unittest.TestCase):
    def test_probability_one_zero(self):
        annotations = [0, 0, 0, 0, 0]
        expected = {0: 1.0, 1: 0.0}
        result = generate_soft_labels_raw_annotations(annotations, range(2))
        self.assertEqual(expected, result)

        annotations = [1, 1, 1, 1, 1]
        expected = {0: 0.0, 1: 1.0}
        result = generate_soft_labels_raw_annotations(annotations, range(2))
        self.assertEqual(expected, result)

    def test_example_case(self):
        annotations = [0, 0, 1, 0, 1]
        expected = {0: 0.6, 1: 0.4}
        result = generate_soft_labels_raw_annotations(annotations, range(2))
        self.assertEqual(expected, result)

    def test_binary_multi_class(self):
        annotations = [0, 0, 0, 1, 1]
        expected = {0: 0.6, 1: 0.4}
        result = generate_soft_labels_raw_annotations(annotations, range(2))
        self.assertEqual(expected, result)

        annotations = [0, 0, 1, 1, 1, 2, 2, 2, 2, 2]
        expected = {0: 0.2, 1: 0.3, 2: 0.5}
        result = generate_soft_labels_raw_annotations(annotations, range(3))
        self.assertEqual(expected, result)

    def test_probability_sum(self):
        annotations = [0, 1, 2, 0, 1, 2, 2]
        result = generate_soft_labels_raw_annotations(annotations, range(3))
        self.assertAlmostEqual(sum(result.values()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
