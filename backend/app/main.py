from __future__ import annotations
import json, os, re
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .algorithms import *
from .schemas import *

load_dotenv()
app = FastAPI(title="Numerical Methods Visualizer API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

METHODS = {
    "gauss-seidel": gauss_seidel, "gauss-jordan": gauss_jordan, "secant": secant,
    "lagrange": lagrange, "cubic-spline": cubic_spline, "curve-fitting": curve_fitting,
    "trapezoidal": trapezoidal, "simpson": simpson, "gaussian-quadrature": gaussian_quadrature,
    "euler": euler, "modified-euler": modified_euler, "rk4": rk4,
    "laplace": laplace, "poisson": poisson, "heat-equation": heat_equation,
}

def classify(text: str):
    lower=text.lower(); groups=[]
    if "laplace" in lower or "∇²u" in lower: groups=["laplace"]
    elif "poisson" in lower or "∇²u" in lower and "=" in lower: groups=["poisson"]
    elif "heat equation" in lower or "∂u/∂t" in lower: groups=["heat-equation"]
    elif "dy/dx" in lower or "d y" in lower: groups=["euler","modified-euler","rk4"]
    elif re.search(r"\b(x|f)\s*\d?\s*[,=].*\b(y|f)", lower): groups=["lagrange","cubic-spline","curve-fitting"]
    elif "integral" in lower or "∫" in lower: groups=["trapezoidal","simpson","gaussian-quadrature"]
    elif "matrix" in lower or "linear system" in lower: groups=["gauss-seidel","gauss-jordan"]
    else: groups=["secant"]
    return groups

@app.get("/api/health")
def health(): return {"status":"ok"}

@app.get("/api/methods")
def methods(): return {"methods": list(METHODS)}

@app.post("/api/parse", response_model=ParseResponse)
def parse_problem(request: ParseRequest):
    compatible=classify(request.text); return {"problem_type": compatible[0], "detected_text": request.text, "compatible_methods": compatible, "confidence":"medium"}

@app.post("/api/recommend", response_model=Recommendation)
def recommend(request: RecommendationRequest):
    methods=request.compatible_methods or classify(request.problem); chosen=methods[0]
    if "rk4" in methods and ("accuracy" in request.problem.lower() or "ode" in request.problem.lower()): chosen="rk4"
    names={k:k.replace("-"," ").title() for k in METHODS}; names["rk4"]="RK4"; names["heat-equation"]="One-Dimensional Heat Equation"
    alternatives=[names[m] for m in methods if m!=chosen]
    fallback={"recommended_method":names[chosen],"reason":"This method is compatible with the detected problem and provides a strong accuracy/convergence trade-off.","alternatives":alternatives}
    key=os.getenv("GROQ_API_KEY")
    if not key: return fallback
    prompt=("Return JSON only with keys recommended_method, reason, alternatives. "
            f"Choose only from {list(names.values())}. Problem: {request.problem}. "
            f"Compatible methods: {[names[m] for m in methods]}")
    try:
        response=httpx.post("https://api.groq.com/openai/v1/chat/completions",headers={"Authorization":f"Bearer {key}"},json={"model":os.getenv("GROQ_MODEL","openai/gpt-oss-20b"),"temperature":0,"response_format":{"type":"json_object"},"messages":[{"role":"user","content":prompt}]},timeout=12)
        response.raise_for_status(); content=response.json()["choices"][0]["message"]["content"]; candidate=Recommendation.model_validate(json.loads(content))
        aliases={"rk4":"RK4","fourth-order runge-kutta":"RK4","fourth order runge-kutta":"RK4"}
        canonical=aliases.get(candidate.recommended_method.lower().strip(),candidate.recommended_method)
        if canonical not in [names[m] for m in methods]: return fallback
        return {**candidate.model_dump(),"recommended_method":canonical}
    except Exception:
        return fallback

@app.post("/api/upload")
async def upload(file: UploadFile=File(...)):
    if file.content_type not in {"image/png","image/jpeg","application/pdf"}: raise HTTPException(400,"Upload a PNG, JPG, JPEG, or PDF file.")
    data=await file.read(); return {"filename":file.filename,"bytes":len(data),"message":"File received. OCR is not enabled in this local build; paste or verify the extracted question in the text box."}

for slug, algorithm in METHODS.items():
    def route(request: SolveRequest, _algorithm=algorithm):
        try: return _algorithm(**request.params)
        except ValueError as exc: raise HTTPException(422, str(exc)) from exc
        except Exception as exc: raise HTTPException(400, f"Unable to solve this problem: {exc}") from exc
    app.add_api_route(f"/api/methods/{slug}", route, methods=["POST"], response_model=dict)
