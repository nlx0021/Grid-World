import os
import imageio
import numpy as np
from sympy import Matrix
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

def shannon_entropy_funcs_factory(check_domain=True):
    """
    Shannon negative entropy on x >= 0:
        psi(x) = x log x - x
        psi'(x) = log x
        (psi')^{-1}(y) = exp(y)

    Boundary behavior:
        psi'(0+) = -inf

    Inverse score domain:
        y in R
    """

    mirror_info = {
        "name": "shannon",
        "psi_prime_zero": -np.inf,
        "psi_prime_one": 0.0,
        "inv_score_lower": -np.inf,
        "inv_score_upper": np.inf,
    }

    def psi(x):
        x = np.asarray(x, dtype=float)

        if check_domain and np.any(x < 0):
            raise ValueError("Shannon entropy is defined for x >= 0.")

        with np.errstate(divide="ignore", invalid="ignore"):
            x_log_x = np.where(x == 0, 0.0, x * np.log(x))

        return x_log_x - x

    def psi_prime(x):
        x = np.asarray(x, dtype=float)

        if check_domain and np.any(x <= 0):
            raise ValueError("Shannon psi_prime requires x > 0.")

        return np.log(x)

    def psi_prime_inv(y):
        y = np.asarray(y, dtype=float)

        with np.errstate(over="ignore"):
            return np.exp(y)

    return psi, psi_prime, psi_prime_inv, mirror_info


def tsallis_entropy_funcs_factory(q, check_domain=True):
    """
    Tsallis-type mirror map on x >= 0:
        psi(x) = (x^q - 1) / (q - 1)

    Derivative:
        psi'(x) = q/(q-1) * x^(q-1)

    Supports:
        q > 1:
            psi'(0+) = 0
            inverse score domain y >= 0
            lower active set exists.

        0 < q < 1:
            psi'(0+) = -inf
            inverse score domain y < 0
            no lower active set.
    """
    assert q > 0 and q != 1, "q must satisfy q > 0 and q != 1."

    if q > 1:
        mirror_info = {
            "name": f"tsallis_q_{q}",
            "psi_prime_zero": 0.0,
            "psi_prime_one": q / (q - 1.0),
            "inv_score_lower": 0.0,
            "inv_score_upper": np.inf,
        }
    else:
        mirror_info = {
            "name": f"tsallis_q_{q}",
            "psi_prime_zero": -np.inf,
            "psi_prime_one": q / (q - 1.0),
            "inv_score_lower": -np.inf,
            "inv_score_upper": 0.0,
        }

    def psi(x):
        x = np.asarray(x, dtype=float)

        if check_domain and np.any(x < 0):
            raise ValueError("Tsallis mirror map here is defined for x >= 0.")

        return (np.power(x, q) - 1.0) / (q - 1.0)

    def psi_prime(x):
        x = np.asarray(x, dtype=float)

        if check_domain:
            if q < 1 and np.any(x <= 0):
                raise ValueError("For 0<q<1, psi_prime requires x > 0.")
            if q > 1 and np.any(x < 0):
                raise ValueError("For q>1, psi_prime expects x >= 0.")

        with np.errstate(divide="ignore", invalid="ignore"):
            return (q / (q - 1.0)) * np.power(x, q - 1.0)

    def psi_prime_inv(y):
        y = np.asarray(y, dtype=float)

        if check_domain:
            if q > 1 and np.any(y < 0):
                raise ValueError("For q>1, psi_prime_inv expects y >= 0.")
            if 0 < q < 1 and np.any(y >= 0):
                raise ValueError("For 0<q<1, psi_prime_inv expects y < 0.")

        base = y * (q - 1.0) / q

        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            return np.power(base, 1.0 / (q - 1.0))

    return psi, psi_prime, psi_prime_inv, mirror_info


def fermi_dirac_entropy_funcs_factory(check_domain=True):
    """
    Fermi-Dirac entropy on x in [0, 1]:
        psi(x) = x log x + (1-x) log(1-x)

    Derivative:
        psi'(x) = log(x / (1-x))

    Inverse derivative:
        (psi')^{-1}(y) = 1 / (1 + exp(-y))

    Boundary behavior:
        psi'(0+) = -inf
        psi'(1-) = +inf

    Inverse score domain:
        y in R
    """

    mirror_info = {
        "name": "fermi_dirac",
        "psi_prime_zero": -np.inf,
        "psi_prime_one": np.inf,
        "inv_score_lower": -np.inf,
        "inv_score_upper": np.inf,
    }

    def psi(x):
        x = np.asarray(x, dtype=float)

        if check_domain and (np.any(x < 0) or np.any(x > 1)):
            raise ValueError("Fermi-Dirac entropy is defined for x in [0, 1].")

        with np.errstate(divide="ignore", invalid="ignore"):
            term1 = np.where(x == 0, 0.0, x * np.log(x))
            term2 = np.where(x == 1, 0.0, (1.0 - x) * np.log(1.0 - x))

        return term1 + term2

    def psi_prime(x):
        x = np.asarray(x, dtype=float)

        if check_domain and (np.any(x < 0) or np.any(x > 1)):
            raise ValueError("Fermi-Dirac psi_prime requires x in (0, 1).")

        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(x / (1.0 - x))

    def psi_prime_inv(y):
        y = np.asarray(y, dtype=float)

        # Numerically stable sigmoid.
        return np.where(
            y >= 0,
            1.0 / (1.0 + np.exp(-y)),
            np.exp(y) / (1.0 + np.exp(y))
        )

    return psi, psi_prime, psi_prime_inv, mirror_info

def hellinger_funcs_factory(check_domain=True):
    """
    Hellinger-type mirror map on probability domain x in [0, 1]:
        psi(x) = -sqrt(1 - x^2)

    Derivative:
        psi'(x) = x / sqrt(1 - x^2)

    Inverse derivative:
        (psi')^{-1}(y) = y / sqrt(1 + y^2)

    Boundary behavior:
        psi'(0+) = 0
        psi'(1-) = +inf

    Lower active set exists because psi'(0+) is finite.
    """

    mirror_info = {
        "name": "hellinger",
        "psi_prime_zero": 0.0,
        "psi_prime_one": np.inf,
        "inv_score_lower": 0.0,
        "inv_score_upper": np.inf,
    }

    def psi(x):
        x = np.asarray(x, dtype=float)

        if check_domain and (np.any(x < 0) or np.any(x > 1)):
            raise ValueError("Hellinger map expects probability x in [0, 1].")

        with np.errstate(invalid="ignore"):
            return -np.sqrt(1.0 - x ** 2)

    def psi_prime(x):
        x = np.asarray(x, dtype=float)

        if check_domain and (np.any(x < 0) or np.any(x >= 1)):
            raise ValueError("Hellinger psi_prime requires x in [0, 1).")

        with np.errstate(divide="ignore", invalid="ignore"):
            return x / np.sqrt(1.0 - x ** 2)

    def psi_prime_inv(y):
        y = np.asarray(y, dtype=float)

        if check_domain and np.any(y < 0):
            raise ValueError("Hellinger psi_prime_inv expects y >= 0.")

        with np.errstate(over="ignore", invalid="ignore"):
            return y / np.sqrt(1.0 + y ** 2)

    return psi, psi_prime, psi_prime_inv, mirror_info