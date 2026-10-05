import pytest
from fastapi.testclient import TestClient

from app.main import app


CASES = [
    ("gauss-seidel", {"A": [[4, 1], [2, 3]], "b": [1, 2], "x0": [0, 0]}),
    ("gauss-jordan", {"A": [[2, 1], [1, 3]], "b": [1, 2]}),
    ("secant", {"equation": "x^2-2", "x0": 1, "x1": 2}),
    ("lagrange", {"points": [[0, 0], [1, 1], [2, 4]], "x": 1.5}),
    ("cubic-spline", {"points": [[0, 0], [1, 1], [2, 0]], "x": 0.5}),
    ("curve-fitting", {"points": [[0, 1], [1, 3], [2, 5]], "model": "linear"}),
    ("trapezoidal", {"equation": "x^2", "a": 0, "b": 1, "n": 8}),
    ("simpson", {"equation": "x^2", "a": 0, "b": 1, "n": 10}),
    ("gaussian-quadrature", {"equation": "x^2", "a": 0, "b": 1, "points": 3}),
    ("euler", {"equation": "x+y", "x0": 0, "y0": 1, "h": 0.1, "steps": 5}),
    ("modified-euler", {"equation": "x+y", "x0": 0, "y0": 1, "h": 0.1, "steps": 5}),
    ("rk4", {"equation": "x+y", "x0": 0, "y0": 1, "h": 0.1, "steps": 5}),
    ("laplace", {"nx": 8, "ny": 8, "boundary": {"top": 1}, "max_iterations": 20}),
    ("poisson", {"source": "1", "nx": 8, "ny": 8, "max_iterations": 20}),
    ("heat-equation", {"alpha": 1, "length": 1, "initial_condition": "sin(pi*x)", "left_boundary": 0, "right_boundary": 0, "dx": 0.1, "dt": 0.002, "total_time": 0.01}),
]


@pytest.mark.parametrize("slug,params", CASES)
def test_method_endpoint_returns_real_result(slug, params):
    response = TestClient(app).post(f"/api/methods/{slug}", json={"params": params})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "success"
    assert "final_result" in payload
    assert payload["iterations"]
    if slug in {"laplace", "poisson"}:
        assert len(payload["metadata"]["snapshots"]) == len(payload["iterations"]) + 1
    if slug == "heat-equation":
        assert len(payload["metadata"]["snapshots"]) == len(payload["iterations"])
