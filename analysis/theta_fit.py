#!/usr/bin/env python3
import argparse

import numpy as np


def fit_theta(N_values, theta_values):
    logN = np.log(N_values)
    logTheta = np.log(theta_values)
    alpha, log_c = np.polyfit(logN, logTheta, 1)
    return -alpha, np.exp(log_c)


def fit_theta_with_stats(N_values, theta_values, z=1.96):
    N_values = np.asarray(N_values, dtype=float)
    theta_values = np.asarray(theta_values, dtype=float)

    if N_values.shape != theta_values.shape:
        raise ValueError("N_values and theta_values must have the same shape")
    if np.any(N_values <= 0) or np.any(theta_values <= 0):
        raise ValueError("N_values and theta_values must be strictly positive")

    logN = np.log(N_values)
    logTheta = np.log(theta_values)

    coeffs, cov = np.polyfit(logN, logTheta, 1, cov=True)
    slope, intercept = coeffs

    pred = slope * logN + intercept
    ss_res = np.sum((logTheta - pred) ** 2)
    ss_tot = np.sum((logTheta - np.mean(logTheta)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0))

    se_slope = float(np.sqrt(max(cov[0, 0], 0.0)))
    se_intercept = float(np.sqrt(max(cov[1, 1], 0.0)))

    alpha = float(-slope)
    c = float(np.exp(intercept))

    alpha_ci = (
        float(-(slope + z * se_slope)),
        float(-(slope - z * se_slope)),
    )
    c_ci = (
        float(np.exp(intercept - z * se_intercept)),
        float(np.exp(intercept + z * se_intercept)),
    )

    return {
        "alpha": alpha,
        "c": c,
        "r2": r2,
        "alpha_ci": alpha_ci,
        "c_ci": c_ci,
    }


def _parse_csv_floats(text):
    return np.asarray([float(x.strip()) for x in text.split(",") if x.strip()], dtype=float)


def main():
    parser = argparse.ArgumentParser(description="Fit theta scaling law: theta = c * N^(-alpha)")
    parser.add_argument("--N", required=True, help="Comma-separated N values, e.g. 0.5,1.5,7")
    parser.add_argument("--theta", required=True, help="Comma-separated theta values")
    args = parser.parse_args()

    N_values = _parse_csv_floats(args.N)
    theta_values = _parse_csv_floats(args.theta)
    stats = fit_theta_with_stats(N_values, theta_values)

    print(f"alpha = {stats['alpha']:.6f}")
    print(f"R^2 = {stats['r2']:.6f}")
    print(f"95% CI(alpha) = [{stats['alpha_ci'][0]:.6f}, {stats['alpha_ci'][1]:.6f}]")
    print(f"95% CI(c) = [{stats['c_ci'][0]:.6f}, {stats['c_ci'][1]:.6f}]")


if __name__ == "__main__":
    main()
