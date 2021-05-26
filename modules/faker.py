from typing import Dict
import numpy as np
from faker import Faker
fake = Faker()


def create_fake_ab_test_results(n: np.int64, prob_a: np.float64) -> Dict[np.array, np.array]:
    results = dict()
    results["person"] = np.array([fake.name() for elem in np.arange(n)])
    results["choice"] = np.random.choice(2, size=n, p=[prob_a, 1-prob_a])
    return results
