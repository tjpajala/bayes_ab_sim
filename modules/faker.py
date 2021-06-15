from typing import Dict
import numpy as np
from faker import Faker
import numpyro.distributions as dist
from jax import random
from time import time_ns
fake = Faker()


def create_fake_ab_test_results(n: np.int64, prob_a: np.float64) -> Dict[np.array, np.array]:
    results = dict()
    results["person"] = np.array([fake.name() for elem in np.arange(n)])
    results["choice"] = np.random.choice(2, size=n, p=[prob_a, 1-prob_a])
    return results


def create_fake_likert_results(n: np.int64, center: np.float64, std: np.float64, diff: np.float64,
                               lowest=0, highest=5,
                               round_responses: np.int64 = None) -> Dict[np.array, np.array]:
    results = dict()
    results["person"] = np.array([fake.name() for elem in np.arange(n)])
    # A location is center + diff, so A is on average better
    results["A"] = truncated_int_dist(n, lowest, highest, base_distribution=dist.Normal(center + diff, std)) #np.random.normal(loc=center + diff, size=n, scale=std)
    results["B"] = truncated_int_dist(n, lowest, highest, base_distribution=dist.Normal(center, std)) #np.random.normal(loc=center, size=n, scale=std)
    if round_responses is not None:
        results["A"] = np.round(results["A"])
        results["B"] = np.round(results["B"])
    return results


def truncated_int_dist(n: int, lowest: int, highest: int, base_distribution: dist.Distribution = None):
    if base_distribution is None:
        mu = lowest + (highest - lowest) / 2
        sd = 1
        base_distribution = dist.Normal(mu, sd)
    trunc = dist.TwoSidedTruncatedDistribution(base_distribution, low=lowest, high=highest)
    d = np.round(trunc.sample(random.PRNGKey(time_ns()), (n,)), 0)
    return d
