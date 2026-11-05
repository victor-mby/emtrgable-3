import unittest
import numpy as np

from config import FEATURES, TARGET
from src.data_generator import generate_dataset, validate_dataset
from src.preprocessing import split_and_scale
from src.clustering import fit_kmeans
from src.dbscan import fit_dbscan
from src.decision_tree import train_and_evaluate


class TestModelosBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate_dataset()
        validate_dataset(cls.df)
        cls.x_train, cls.x_test, cls.y_train, cls.y_test, cls.scaler = split_and_scale(cls.df)
        cls.x_all = cls.scaler.transform(cls.df[FEATURES].astype(float))

    def test_dataset(self):
        self.assertEqual(self.df.shape[0], 250)
        self.assertEqual(self.df[TARGET].sum(), 50)
        self.assertFalse(self.df[FEATURES].isna().any().any())

    def test_kmeans(self):
        _, labels = fit_kmeans(self.x_all)
        self.assertEqual(len(labels), 250)
        self.assertGreaterEqual(len(set(labels)), 2)

    def test_dbscan(self):
        _, labels, info = fit_dbscan(self.x_all)
        self.assertEqual(len(labels), 250)
        self.assertIn("grupos", info)
        self.assertIn("ruido", info)

    def test_decision_tree(self):
        _, pred, metrics = train_and_evaluate(
            self.x_train, self.y_train, self.x_test, self.y_test
        )
        self.assertEqual(len(pred), len(self.y_test))
        self.assertIn("f1_score", metrics)
        self.assertTrue(0 <= metrics["f1_score"] <= 1)


if __name__ == "__main__":
    unittest.main()
