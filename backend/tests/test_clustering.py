"""
Tests for backend/clustering.py

Covers the perform_global_clustering() function which groups embeddings
using K-Means for hierarchical RAG.
"""

import unittest
import numpy as np

from backend.clustering import perform_global_clustering


class TestPerformGlobalClustering(unittest.TestCase):
    """Unit tests for perform_global_clustering()."""

    def test_empty_input_returns_empty_dict(self):
        """Empty embedding list returns empty dict."""
        result = perform_global_clustering([])
        self.assertEqual(result, {})

    def test_single_item_returns_single_cluster(self):
        """Single embedding is placed into cluster 0."""
        embeddings = [[0.1, 0.2, 0.3]]
        result = perform_global_clustering(embeddings)
        self.assertEqual(result, {0: [0]})

    def test_small_list_below_threshold_returns_single_cluster(self):
        """When n_samples <= max_cluster_size, all items go into cluster 0."""
        embeddings = [[float(i), float(i + 1)] for i in range(5)]
        result = perform_global_clustering(embeddings, max_cluster_size=20)
        self.assertIn(0, result)
        self.assertEqual(sorted(result[0]), list(range(5)))

    def test_large_list_produces_multiple_clusters(self):
        """More than max_cluster_size items produce multiple clusters."""
        np.random.seed(42)
        embeddings = np.random.rand(50, 4).tolist()
        result = perform_global_clustering(embeddings, max_cluster_size=10)
        self.assertGreater(len(result), 1)

    def test_all_indices_present_in_clusters(self):
        """Every embedding index appears exactly once across all clusters."""
        np.random.seed(0)
        n = 40
        embeddings = np.random.rand(n, 8).tolist()
        result = perform_global_clustering(embeddings, max_cluster_size=10)

        all_indices = []
        for indices in result.values():
            all_indices.extend(indices)

        self.assertEqual(sorted(all_indices), list(range(n)))

    def test_cluster_values_are_lists_of_ints(self):
        """Cluster values must be lists of Python ints (not numpy int types)."""
        np.random.seed(1)
        embeddings = np.random.rand(30, 4).tolist()
        result = perform_global_clustering(embeddings, max_cluster_size=8)

        for cluster_id, indices in result.items():
            self.assertIsInstance(cluster_id, int)
            for idx in indices:
                self.assertIsInstance(idx, int)

    def test_cluster_id_is_int_not_numpy(self):
        """Cluster keys are plain Python ints (explicit conversion check)."""
        np.random.seed(7)
        embeddings = np.random.rand(25, 3).tolist()
        result = perform_global_clustering(embeddings, max_cluster_size=5)
        for key in result.keys():
            self.assertNotIsInstance(key, np.integer,
                                     msg="Cluster keys must be Python int, not numpy int")

    def test_max_cluster_size_one_creates_many_clusters(self):
        """max_cluster_size=1 forces roughly n clusters for n items."""
        np.random.seed(99)
        n = 10
        embeddings = np.random.rand(n, 2).tolist()
        result = perform_global_clustering(embeddings, max_cluster_size=1)
        self.assertGreater(len(result), 1)

    def test_returns_dict(self):
        """Return type is always a dict."""
        result = perform_global_clustering([[0.5, 0.5]])
        self.assertIsInstance(result, dict)

    def test_two_well_separated_groups_cluster_together(self):
        """Two clearly separated point clouds end up in separate clusters."""
        group_a = [[0.0, 0.0]] * 12
        group_b = [[100.0, 100.0]] * 12
        embeddings = group_a + group_b
        result = perform_global_clustering(embeddings, max_cluster_size=10)

        cluster_sets = [set(v) for v in result.values()]
        a_indices = set(range(12))
        b_indices = set(range(12, 24))

        # At least one cluster should contain only A-group or only B-group
        clean_separation = any(
            (c.issubset(a_indices) and len(c) > 0) or
            (c.issubset(b_indices) and len(c) > 0)
            for c in cluster_sets
        )
        self.assertTrue(clean_separation, "Well-separated groups should cluster apart")


if __name__ == "__main__":
    unittest.main()
