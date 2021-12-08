"""
we generate data to represent behaviour
"""

import random
import string
from typing import Tuple

from scipy.stats import skewnorm


def get_skewed_values():
    return int(abs(skewnorm.rvs(10) * 6 + 15))


def get_behavioural_sequence() -> Tuple[str, int]:
    """Generates a random alphanumeric string representing some generic behavioural sequence.
    :return: A tuple, first element is the behaviour string, the second is the "outcome", some interesting behaviour
    """

    alpha_numeric_string = string.ascii_lowercase + string.digits

    if random.uniform(0, 1) < 0.9:
        return (
            "".join(
                random.choice(alpha_numeric_string)
                for _ in range(random.randint(10, 25))
            ),
            0,
        )

    alpha_len = len(alpha_numeric_string) - 1
    return (
        "".join(
            alpha_numeric_string[min(get_skewed_values(), alpha_len)]
            for _ in range(random.randint(10, 25))
        ),
        1,
    )
