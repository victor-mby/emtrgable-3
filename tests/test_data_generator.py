"""Pruebas básicas del generador de datos."""

import unittest

from config import FEATURES, METADATA, TARGET
from src.data_generator import generate_dataset, validate_dataset


class DataGeneratorTests(unittest.TestCase):
    def test_shape_and_columns(self):
        df = generate_dataset()
        self.assertEqual(len(df), 250)
        self.assertEqual(list(df.columns), METADATA + FEATURES + [TARGET])

    def test_class_distribution(self):
        df = generate_dataset()
        self.assertEqual(int((df[TARGET] == 0).sum()), 200)
        self.assertEqual(int((df[TARGET] == 1).sum()), 50)

    def test_validation(self):
        validate_dataset(generate_dataset())


if __name__ == "__main__":
    unittest.main()
