"""
we generate data to represent behaviour
"""

import random
import string
from typing import Tuple

from scipy.stats import skewnorm


def get_skewed_index():
    return int(abs(skewnorm.rvs(10) * 6 + 15))


def get_skewed_random_string(str_seq: str, min_len: int = 10, max_len: int = 25) -> str:
    alpha_len = len(str_seq) - 1
    return "".join(
        str_seq[min(get_skewed_index(), alpha_len)]
        for _ in range(random.randint(min_len, max_len))
    )


def get_unif_random_string(str_seq: str, min_len: int = 10, max_len: int = 25) -> str:
    return "".join(
        random.choice(str_seq) for _ in range(random.randint(min_len, max_len))
    )


def get_behavioural_sequence() -> Tuple[str, int]:
    """Generates a random alphanumeric string representing some generic behavioural sequence.
    :return: A tuple, first element is the behaviour string, the second is the "outcome", some interesting behaviour
    """

    alpha_numeric_string = string.ascii_lowercase + string.digits

    if random.uniform(0, 1) < 0.9:
        return get_unif_random_string(alpha_numeric_string), 0

    return get_skewed_random_string(alpha_numeric_string), 1
