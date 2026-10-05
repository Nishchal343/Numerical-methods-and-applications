import numpy as np
import pytest
from app.algorithms import *

def test_gauss_seidel():
    r=gauss_seidel([[4,1],[2,3]],[1,2],max_iterations=100); assert np.allclose(r["final_result"],[.1,.6],atol=1e-6)
def test_gauss_jordan(): assert np.allclose(gauss_jordan([[2,1],[1,3]],[1,2])["final_result"],[.2,.6])
def test_secant(): assert abs(secant("x**2-2",1,2)["final_result"]-np.sqrt(2))<1e-7
def test_interpolation_and_spline():
    assert abs(lagrange([[0,0],[1,1],[2,4]],1.5)["final_result"]-2.25)<1e-8
    assert isinstance(cubic_spline([[0,0],[1,1],[2,0]],.5)["final_result"],float)
def test_curve_fitting(): assert curve_fitting([[0,1],[1,3],[2,5]],"linear")["final_result"]["r2"]>.99
def test_integration():
    assert abs(trapezoidal("x**2",0,1,100)["final_result"]-1/3)<1e-4
    assert abs(simpson("x**2",0,1,10)["final_result"]-1/3)<1e-10
    assert abs(gaussian_quadrature("x**2",0,1,3)["final_result"]-1/3)<1e-10
def test_odes():
    assert abs(euler("y",0,1,.01,10)["final_result"]-np.exp(.1))<.01
    assert abs(modified_euler("y",0,1,.01,10)["final_result"]-np.exp(.1))<.001
    assert abs(rk4("y",0,1,.01,10)["final_result"]-np.exp(.1))<1e-7
def test_pdes():
    assert len(laplace(8,8)["final_result"])==8
    assert len(poisson("1",8,8)["final_result"])==8
    assert heat_equation(1,.2,"x",0,0,.02,.0001,.001)["metadata"]["stability_ratio"]<=.5
def test_validation():
    with pytest.raises(ValueError): simpson("x",0,1,3)
    with pytest.raises(ValueError): heat_equation(1,1,"x",0,0,.1,.01,1)
