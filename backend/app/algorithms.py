"""Pure numerical algorithms. No web framework or user-code execution belongs here."""
from __future__ import annotations

import json
import math
from typing import Any, Callable

import numpy as np
import sympy as sp


def parse_expr(text: str, variables: tuple[str, ...] = ("x",)) -> tuple[sp.Expr, Callable[..., float]]:
    normalized = str(text).strip().replace("^", "**").replace("ln", "log")
    syms = sp.symbols(" ".join(variables))
    if len(variables) == 1:
        syms = (syms,) if not isinstance(syms, tuple) else syms
    try:
        expr = sp.sympify(normalized, locals={"e": sp.E, "pi": sp.pi, "ln": sp.log})
        allowed = set(syms)
        if not expr.free_symbols.issubset(allowed):
            raise ValueError(f"Use only these variables: {', '.join(variables)}")
        fn = sp.lambdify(syms, expr, modules=["numpy", "math"])
        return expr, fn
    except Exception as exc:
        raise ValueError("Unable to parse the equation. Please check the syntax.") from exc


def _json(value: Any) -> Any:
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.floating, np.integer)): return value.item()
    return value


def gauss_seidel(A, b, x0=None, tolerance=1e-8, max_iterations=100):
    A, b = np.asarray(A, float), np.asarray(b, float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or len(b) != A.shape[0]: raise ValueError("A must be square and match b.")
    if np.any(np.isclose(np.diag(A), 0)): raise ValueError("The matrix contains a zero diagonal.")
    x = np.zeros(len(b)) if x0 is None else np.asarray(x0, float).copy()
    if len(x) != len(b): raise ValueError("Initial guess has the wrong dimension.")
    dominant = bool(np.all(np.abs(np.diag(A)) >= np.sum(np.abs(A), axis=1) - np.abs(np.diag(A))))
    rows, converged = [], False
    for iteration in range(1, max_iterations + 1):
        old = x.copy()
        for i in range(len(b)):
            x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
        error = float(np.linalg.norm(x - old, ord=np.inf)); residual = float(np.linalg.norm(A @ x - b, ord=np.inf))
        rows.append({"iteration": iteration, "x": _json(x.copy()), "error": error, "residual": residual})
        if error <= tolerance: converged = True; break
    return {"method": "Gauss-Seidel", "status": "success", "final_result": _json(x), "iterations": rows, "error": rows[-1]["error"] if rows else None, "metadata": {"converged": converged, "diagonally_dominant": dominant}}


def gauss_jordan(A, b):
    aug = np.column_stack([np.asarray(A, float), np.asarray(b, float)])
    m, n = aug.shape[0], aug.shape[1] - 1; operations = []
    for col in range(n):
        pivot = col + int(np.argmax(np.abs(aug[col:, col])))
        if abs(aug[pivot, col]) < 1e-12: continue
        if pivot != col:
            aug[[col, pivot]] = aug[[pivot, col]]; operations.append({"operation": f"R{col+1} ↔ R{pivot+1}", "matrix": _json(aug.copy()), "value": float(np.linalg.norm(aug))})
        aug[col] /= aug[col, col]; operations.append({"operation": f"R{col+1} → R{col+1} / pivot", "matrix": _json(aug.copy()), "value": float(np.linalg.norm(aug))})
        for row in range(m):
            if row == col: continue
            factor = aug[row, col]
            if abs(factor) > 1e-12:
                aug[row] -= factor * aug[col]; operations.append({"operation": f"R{row+1} → R{row+1} - ({factor:.6g})R{col+1}", "matrix": _json(aug.copy()), "value": float(np.linalg.norm(aug))})
    ranks = np.linalg.matrix_rank(np.asarray(A, float)); rank_aug = np.linalg.matrix_rank(aug)
    if rank_aug > ranks: status = "inconsistent"; result = None
    elif ranks < n: status = "singular"; result = None
    else: status = "success"; result = _json(aug[:, -1])
    return {"method": "Gauss-Jordan", "status": status, "final_result": result, "iterations": operations, "metadata": {"rank": int(ranks), "augmented_rank": int(rank_aug)}}


def secant(equation, x0, x1, tolerance=1e-8, max_iterations=100):
    _, f = parse_expr(equation); rows = []; converged = False
    for i in range(1, max_iterations + 1):
        f0, f1 = float(f(x0)), float(f(x1)); den = f1 - f0
        if abs(den) < 1e-14: raise ValueError("Division by a near-zero denominator occurred. Try different initial guesses.")
        x2 = x1 - f1 * (x1 - x0) / den; error = abs(x2 - x1)
        rows.append({"iteration": i, "x0": x0, "x1": x1, "f_x1": f1, "x_next": x2, "error": error})
        x0, x1 = x1, x2
        if error <= tolerance: converged = True; break
    return {"method": "Secant", "status": "success", "final_result": x1, "iterations": rows, "error": rows[-1]["error"] if rows else None, "metadata": {"converged": converged, "equation": equation}}


def lagrange(points, x):
    pts = np.asarray(points, float); xs, ys = pts[:, 0], pts[:, 1]
    if len(set(xs)) != len(xs): raise ValueError("Interpolation x values must be distinct.")
    t = sp.symbols("x"); poly = 0; basis = []
    for i in range(len(xs)):
        li = sp.prod((t - xs[j]) / (xs[i] - xs[j]) for j in range(len(xs)) if i != j)
        basis.append({"index": i, "basis": str(sp.expand(li)), "value": float(li.subs(t, x))}); poly += ys[i] * li
    val = float(poly.subs(t, x))
    return {"method": "Lagrange Interpolation", "status": "success", "final_result": val, "iterations": basis, "metadata": {"polynomial": str(sp.expand(poly)), "points": _json(pts)}}


def cubic_spline(points, x, boundary="natural", derivatives=None):
    pts = np.asarray(points, float); xs, ys = pts[:, 0], pts[:, 1]
    if np.any(np.diff(xs) <= 0): raise ValueError("Spline x values must be strictly increasing.")
    n = len(xs) - 1; h = np.diff(xs); A = np.zeros((n + 1, n + 1)); rhs = np.zeros(n + 1)
    if boundary == "clamped" and derivatives:
        A[0, 0] = 2 * h[0]; A[0, 1] = h[0]; rhs[0] = 3 * ((ys[1]-ys[0])/h[0] - derivatives[0])
        A[n, n-1] = h[-1]; A[n, n] = 2*h[-1]; rhs[n] = 3 * (derivatives[1] - (ys[-1]-ys[-2])/h[-1])
    else: A[0, 0] = A[n, n] = 1
    for i in range(1, n): A[i, i-1:i+2] = [h[i-1], 2*(h[i-1]+h[i]), h[i]]; rhs[i] = 3*((ys[i+1]-ys[i])/h[i] - (ys[i]-ys[i-1])/h[i-1])
    c = np.linalg.solve(A, rhs); coeffs = []
    for i in range(n):
        a = ys[i]; b = (ys[i+1]-ys[i])/h[i] - h[i]*(2*c[i]+c[i+1])/3; d = (c[i+1]-c[i])/(3*h[i]); coeffs.append([a,b,c[i],d])
    idx = min(max(int(np.searchsorted(xs, x) - 1), 0), n-1); dx = x - xs[idx]; a,b,cc,d = coeffs[idx]
    val = a+b*dx+cc*dx**2+d*dx**3
    return {"method": "Cubic Spline", "status": "success", "final_result": float(val), "iterations": [{"segment": i, "interval": [xs[i], xs[i+1]], "coefficients": coeffs[i], "value": float(np.linalg.norm(coeffs[i]))} for i in range(n)], "metadata": {"boundary": boundary, "points": _json(pts)}}


def curve_fitting(points, model="linear", degree=2):
    pts = np.asarray(points, float); x, y = pts[:, 0], pts[:, 1]
    if model == "linear": design, labels = np.column_stack([x, np.ones(len(x))]), ["a", "b"]
    elif model == "polynomial": design, labels = np.column_stack([x**i for i in range(degree, -1, -1)]), [f"a{i}" for i in range(degree, -1, -1)]
    elif model == "exponential":
        if np.any(y <= 0): raise ValueError("Exponential fitting requires positive y values.")
        design, labels = np.column_stack([x, np.ones(len(x))]), ["b", "ln_a"]; coef = np.linalg.lstsq(design, np.log(y), rcond=None)[0]; params = [float(np.exp(coef[1])), float(coef[0])]
    elif model == "logarithmic":
        if np.any(x <= 0): raise ValueError("Logarithmic fitting requires positive x values.")
        design, labels = np.column_stack([np.log(x), np.ones(len(x))]), ["a", "b"]
    elif model == "power":
        if np.any(x <= 0) or np.any(y <= 0): raise ValueError("Power fitting requires positive x and y values.")
        design, labels = np.column_stack([np.log(x), np.ones(len(x))]), ["b", "ln_a"]; coef = np.linalg.lstsq(design, np.log(y), rcond=None)[0]; params = [float(np.exp(coef[1])), float(coef[0])]
    else: raise ValueError("Unknown curve model.")
    if model not in ("exponential", "power"): params = [float(v) for v in np.linalg.lstsq(design, y if model != "exponential" else np.log(y), rcond=None)[0]]
    def predict(z):
        if model == "linear": return params[0]*z+params[1]
        if model == "polynomial": return np.polyval(params, z)
        if model == "logarithmic": return params[0]*np.log(z)+params[1]
        if model == "exponential": return params[0]*np.exp(params[1]*z)
        return params[0]*z**params[1]
    fitted = np.asarray(predict(x)); residuals = y-fitted; sse=float(np.sum(residuals**2)); mse=sse/len(y); r2=1-sse/np.sum((y-y.mean())**2)
    return {"method": "Curve Fitting", "status": "success", "final_result": {"coefficients": params, "r2": float(r2), "sse": sse, "mse": mse, "rmse": math.sqrt(mse)}, "iterations": [{"x": float(a), "y": float(b), "fitted": float(c), "residual": float(d)} for a,b,c,d in zip(x,y,fitted,residuals)], "metadata": {"model": model}}


def _integrand(equation): return parse_expr(equation)[1]
def trapezoidal(equation, a, b, n=1):
    if n < 1: raise ValueError("n must be at least 1.")
    f=_integrand(equation); h=(b-a)/n; xs=np.linspace(a,b,n+1); ys=np.asarray(f(xs),float); contributions=[h*(ys[i]+ys[i+1])/2 for i in range(n)]
    exact=None
    try: exact=float(sp.integrate(parse_expr(equation)[0], (sp.symbols('x'),a,b)))
    except Exception: pass
    result=float(sum(contributions)); return {"method":"Trapezoidal Rule","status":"success","final_result":result,"iterations":[{"i":i,"x0":float(xs[i]),"x1":float(xs[i+1]),"f0":float(ys[i]),"f1":float(ys[i+1]),"contribution":float(contributions[i])} for i in range(n)],"error":abs(result-exact) if exact is not None else None,"metadata":{"h":h,"exact":exact}}


def simpson(equation,a,b,n=2):
    if n < 2 or n % 2: raise ValueError("Simpson's 1/3 rule requires an even number of intervals.")
    f=_integrand(equation); h=(b-a)/n; xs=np.linspace(a,b,n+1); ys=np.asarray(f(xs),float); rows=[]
    for i,(xx,yy) in enumerate(zip(xs,ys)): rows.append({"i":i,"x":float(xx),"fx":float(yy),"coefficient":1 if i in (0,n) else (4 if i%2 else 2),"contribution":float((1 if i in (0,n) else (4 if i%2 else 2))*yy*h/3)})
    result=float(sum(r["contribution"] for r in rows)); return {"method":"Simpson's 1/3 Rule","status":"success","final_result":result,"iterations":rows,"metadata":{"h":h}}


def gaussian_quadrature(equation,a,b,points=3):
    if points not in (2,3,4): raise ValueError("Gaussian quadrature supports 2, 3, or 4 points.")
    nodes,weights=np.polynomial.legendre.leggauss(points); f=_integrand(equation); mapped=(b-a)/2*nodes+(a+b)/2; vals=np.asarray(f(mapped),float); contrib=(b-a)/2*weights*vals; result=float(np.sum(contrib))
    return {"method":"Gaussian Quadrature","status":"success","final_result":result,"iterations":[{"node":float(x),"weight":float(w),"f_node":float(v),"contribution":float(c)} for x,w,v,c in zip(mapped,weights,vals,contrib)],"metadata":{"reference_nodes":_json(nodes),"interval":[a,b]}}


def _ode_rows(equation,x0,y0,h,steps,kind):
    _, f=parse_expr(equation,("x","y")); x=float(x0); y=float(y0); rows=[{"iteration":0,"x":x,"y":y,"error":0.0}]
    for i in range(1,steps+1):
        k1=float(f(x,y))
        if kind=="euler": yn=y+h*k1; row={"iteration":i,"x":x,"y":y,"slope":k1,"next_y":yn,"error":abs(yn-y)}
        elif kind=="modified-euler":
            pred=y+h*k1; k2=float(f(x+h,pred)); yn=y+h*(k1+k2)/2; row={"iteration":i,"x":x,"y":y,"initial_slope":k1,"predicted_y":pred,"predicted_slope":k2,"next_y":yn,"error":abs(yn-pred)}
        else:
            k2=float(f(x+h/2,y+h*k1/2)); k3=float(f(x+h/2,y+h*k2/2)); k4=float(f(x+h,y+h*k3)); yn=y+h*(k1+2*k2+2*k3+k4)/6; row={"iteration":i,"x":x,"y":y,"k1":k1,"k2":k2,"k3":k3,"k4":k4,"next_y":yn,"error":abs(yn-y)}
        x += h; y=yn; row["next_x"]=x; rows.append(row)
    return rows
def euler(equation,x0,y0,h,steps): return {"method":"Euler's Method","status":"success","final_result":_ode_rows(equation,x0,y0,h,steps,"euler")[-1]["next_y"],"iterations":_ode_rows(equation,x0,y0,h,steps,"euler"),"metadata":{"equation":equation}}
def modified_euler(equation,x0,y0,h,steps):
    rows=_ode_rows(equation,x0,y0,h,steps,"modified-euler"); return {"method":"Modified Euler","status":"success","final_result":rows[-1]["next_y"],"iterations":rows,"metadata":{"equation":equation}}
def rk4(equation,x0,y0,h,steps):
    rows=_ode_rows(equation,x0,y0,h,steps,"rk4"); return {"method":"Fourth-Order Runge-Kutta","status":"success","final_result":rows[-1]["next_y"],"iterations":rows,"metadata":{"equation":equation}}


def _grid_solve(source, nx, ny, boundary, tolerance, max_iterations):
    u=np.zeros((ny,nx),float); u[0,:]=boundary.get("bottom",0); u[-1,:]=boundary.get("top",0); u[:,0]=boundary.get("left",0); u[:,-1]=boundary.get("right",0); snapshots=[_json(u.copy())]; history=[]
    hx=1/(nx-1); hy=1/(ny-1); inv_hx2=1/(hx*hx); inv_hy2=1/(hy*hy); denominator=2*(inv_hx2+inv_hy2)
    for it in range(1,max_iterations+1):
        old=u.copy()
        for j in range(1,ny-1):
            for i in range(1,nx-1): u[j,i]=((u[j,i-1]+u[j,i+1])*inv_hx2+(u[j-1,i]+u[j+1,i])*inv_hy2-source[j,i])/denominator
        err=float(np.max(np.abs(u-old))); history.append({"iteration":it,"error":err});
        # Keep every relaxation state so the frontend can scrub the actual
        # numerical surface instead of interpolating or showing only the end.
        snapshots.append(_json(u.copy()))
        if err <= tolerance: break
    return u,history,snapshots
def _normalise_boundary(boundary):
    if boundary is None or boundary == "": return {}
    if isinstance(boundary, str):
        try: boundary=json.loads(boundary)
        except json.JSONDecodeError as exc: raise ValueError("Boundary conditions must be valid JSON, for example {\"top\": 1, \"bottom\": 0}.") from exc
    if not isinstance(boundary, dict): raise ValueError("Boundary conditions must be an object with top, bottom, left, and right values.")
    try: return {str(k): float(v) for k,v in boundary.items()}
    except (TypeError, ValueError) as exc: raise ValueError("Boundary values must be numeric.") from exc
def _validate_grid(nx,ny):
    if int(nx) != nx or int(ny) != ny or nx < 3 or ny < 3: raise ValueError("PDE grids must have at least 3 rows and 3 columns.")
def laplace(nx=25,ny=25,boundary=None,tolerance=1e-5,max_iterations=1000):
    _validate_grid(nx,ny); boundary=_normalise_boundary(boundary); boundary=boundary or {"top":1}; u,h,s=_grid_solve(np.zeros((ny,nx)),nx,ny,boundary,tolerance,max_iterations); return {"method":"Laplace Equation","status":"success","final_result":_json(u),"iterations":h,"error":h[-1]["error"],"metadata":{"snapshots":s,"converged":h[-1]["error"]<=tolerance}}
def poisson(source,nx=25,ny=25,boundary=None,tolerance=1e-5,max_iterations=1000):
    _validate_grid(nx,ny); boundary=_normalise_boundary(boundary); _,src=_parse_grid_source(source,nx,ny); u,h,s=_grid_solve(src,nx,ny,boundary,tolerance,max_iterations); return {"method":"Poisson Equation","status":"success","final_result":_json(u),"iterations":h,"error":h[-1]["error"],"metadata":{"snapshots":s,"converged":h[-1]["error"]<=tolerance}}
def _parse_grid_source(source,nx,ny):
    _,f=parse_expr(source,("x","y")); xx=np.linspace(0,1,nx); yy=np.linspace(0,1,ny); X,Y=np.meshgrid(xx,yy); return None,np.broadcast_to(np.asarray(f(X,Y),float),(ny,nx))
def heat_equation(alpha,length,initial_condition,left_boundary,right_boundary,dx,dt,total_time):
    r=alpha*dt/dx**2
    if r>0.5: raise ValueError(f"The selected dt/dx values violate the stability condition: α·dt/dx² = {r:.4g} > 0.5.")
    nx=int(round(length/dx))+1; nt=int(round(total_time/dt))+1
    _,fn=parse_expr(initial_condition); x=np.linspace(0,length,nx); values=np.zeros((nt,nx)); values[0]=np.asarray(fn(x),float); values[:,0]=left_boundary; values[:,-1]=right_boundary
    for k in range(nt-1): values[k+1,1:-1]=values[k,1:-1]+r*(values[k,2:]-2*values[k,1:-1]+values[k,:-2])
    rows=[{"time":float(i*dt),"maximum":float(np.max(values[i]))} for i in range(nt)]
    return {"method":"One-Dimensional Heat Equation","status":"success","final_result":_json(values[-1]),"iterations":rows,"metadata":{"x":_json(x),"times":_json(np.arange(nt)*dt),"surface":_json(values),"snapshots":[_json(values[:i+1]) for i in range(nt)],"stability_ratio":r}}
