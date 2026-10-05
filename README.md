# Numerical Methods Visualizer

A local-first educational visualizer for fifteen numerical methods. The backend contains the numerical implementations and a typed FastAPI boundary; the frontend provides dynamic forms, iteration tables, Plotly charts, and a Three.js surface for PDE results.

## Run

### Backend (Python 3.10+)

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Optional `.env`:

```text
GROQ_API_KEY=your_key
GROQ_MODEL=llama-3.1-8b-instant
```

### Frontend (Node 18+)

```powershell
cd frontend
npm install
npm run dev
```

The frontend expects `http://localhost:8000`; set `VITE_API_URL` to override it.

## Tests

```powershell
cd backend
pytest
```

All calculation paths are in `backend/app/algorithms.py` and can be used without starting FastAPI.
