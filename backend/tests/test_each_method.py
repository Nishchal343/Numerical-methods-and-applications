import numpy as np

from app.algorithms import (
    cubic_spline, curve_fitting, euler, gauss_jordan, gauss_seidel,
    gaussian_quadrature, heat_equation, lagrange, laplace, modified_euler,
    poisson, rk4, secant, simpson, trapezoidal,
)


def test_gauss_seidel_known_solution():
    result = gauss_seidel([[4, 1], [2, 3]], [1, 2], [0, 0])
    assert np.allclose(result["final_result"], [0.1, 0.6], atol=1e-7)


def test_gauss_jordan_known_solution():
    assert np.allclose(gauss_jordan([[2, 1], [1, 3]], [1, 2])["final_result"], [0.2, 0.6])


def test_secant_known_root():
    assert abs(secant("x^2 - 2", 1, 2)["final_result"] - np.sqrt(2)) < 1e-7


def test_lagrange_known_polynomial():
    assert abs(lagrange([[0, 0], [1, 1], [2, 4]], 1.5)["final_result"] - 2.25) < 1e-10


def test_cubic_spline_interpolates_data():
    result = cubic_spline([[0, 0], [1, 1], [2, 0]], 1)
    assert abs(result["final_result"] - 1) < 1e-10


def test_curve_fit_linear_model():
    result = curve_fitting([[0, 1], [1, 3], [2, 5]], "linear")["final_result"]
    assert np.allclose(result["coefficients"], [2, 1], atol=1e-10)
    assert result["r2"] > 0.999999


def test_trapezoidal_integral():
    assert abs(trapezoidal("x^2", 0, 1, 100)["final_result"] - 1 / 3) < 1e-4


def test_simpson_integral():
    assert abs(simpson("x^2", 0, 1, 10)["final_result"] - 1 / 3) < 1e-10


def test_gaussian_quadrature_integral():
    assert abs(gaussian_quadrature("x^2", 0, 1, 3)["final_result"] - 1 / 3) < 1e-10


def test_euler_ode():
    assert abs(euler("y", 0, 1, 0.1, 5)["final_result"] - 1.61051) < 1e-8


def test_modified_euler_ode():
    assert abs(modified_euler("y", 0, 1, 0.1, 5)["final_result"] - 1.648721) < 0.002


def test_rk4_ode():
    assert abs(rk4("y", 0, 1, 0.1, 5)["final_result"] - np.exp(0.5)) < 1e-6


def test_laplace_zero_boundary_is_zero():
    result = laplace(10, 10, {"top": 0, "bottom": 0, "left": 0, "right": 0})
    assert np.max(np.abs(result["final_result"])) == 0
    assert len(result["metadata"]["snapshots"]) == len(result["iterations"]) + 1


def test_poisson_source_solution_has_correct_sign_and_residual():
    result = poisson(1, 12, 12, '{"top": 0, "bottom": 0, "left": 0, "right": 0}', tolerance=1e-6)
    grid = np.asarray(result["final_result"])
    assert grid[grid.shape[0] // 2, grid.shape[1] // 2] < 0
    assert result["error"] < 1e-5


def test_heat_equation_matches_analytical_decay():
    result = heat_equation(1, 1, "sin(pi*x)", 0, 0, 0.05, 0.0005, 0.05)
    numerical = np.asarray(result["final_result"])[1:-1]
    x = np.linspace(0, 1, 21)[1:-1]
    exact = np.sin(np.pi * x) * np.exp(-np.pi**2 * 0.05)
    assert np.max(np.abs(numerical - exact)) < 0.01
    assert len(result["metadata"]["snapshots"]) == len(result["iterations"])
