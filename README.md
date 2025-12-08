# GL Reconciliation Web App

FastAPI + pandas web app that ingests GL (File A), Subledger/Source (File B), and an account mapping matrix, reconciles them with tolerance rules, and produces a polished Excel workbook (dashboard, pivots, exception tabs) plus on-page summary.

## Features
- Upload GL, Subledger, and Mapping (CSV/XLSX). Handles missing/extra columns gracefully and reports issues instead of failing.
- Tolerance-aware matching with `mapping_ok`, status classification (EXACT/NEAR/MAPPING_ISSUE/MISMATCH/UNMATCHED_A/UNMATCHED_B), and validation guard for mapping/status conflicts.
- Rich Excel (`gl_recon_report_*.xlsx`): dashboard with dropdown filters + charts, overview, status counts, matches with conditional formatting, unmatched tabs, mismatched amounts, exceptions pivots, mapping copy, missing-map diagnostics, and list sheet for validation.
- Sleek frontend with live KPIs, status breakdown, issues list, and download link.

## Project layout
- `app/main.py` — FastAPI app, upload endpoint, download handler, static site.
- `app/recon.py` — reconciliation logic, validation, status assignment, pivots.
- `app/excel_report.py` — workbook builder (xlsxwriter), dashboard formulas/charts.
- `app/config.py` — defaults and parameter parsing.
- `app/static/index.html` — single-page UI.
- `requirements.txt` — dependencies.
- `reports/` — generated workbooks (gitignored).

## Quickstart (local)
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Open http://localhost:8000 and drop your GL/Subledger/Mapping files. Download link appears after a run (`/api/download/{token}`).

## API
- `POST /api/reconcile` (multipart/form-data):
  - `gl_file`, `src_file`, `mapping_file`: files (csv/xlsx)
  - optional: `amount_tol`, `date_tol`, `date_from`, `date_to`
- `GET /api/download/{token}`: returns XLSX for a previously completed run.

## Deploying to Render
1. Create a new Web Service from this repo.
2. Runtime: Python. Build command: `pip install -r requirements.txt`
3. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Add `PYTHON_VERSION` if needed (e.g., `3.11`).

## Notes
- Files are processed in-memory; generated reports are stored under `reports/` with random tokens.
- Mapping blanks are treated as missing; `mapping_ok` is a strict boolean comparison after string normalization (no numeric coercion).
- Dashboard formulas respect filter dropdowns (GL, Source, Date From/To) using `COUNTIFS` against the `ModelTable`.
