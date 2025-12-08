import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from dotenv import load_dotenv

from .config import ReconParams
from .excel_report import build_excel_report
from .recon import run_reconciliation
from .report_builder import LLMUnavailable, build_llm_report

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="GL Reconciliation", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

report_registry = {}


@app.get("/", response_class=HTMLResponse)
async def index():
    index_file = static_dir / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend not built")
    return index_file.read_text(encoding="utf-8")


@app.post("/api/reconcile")
async def reconcile(
    gl_file: UploadFile = File(..., description="GL file (xlsx or csv)"),
    src_file: UploadFile = File(..., description="Subledger/Source file (xlsx or csv)"),
    mapping_file: UploadFile = File(..., description="Mapping matrix (xlsx or csv)"),
    amount_tol: Optional[str] = Form(None),
    date_tol: Optional[str] = Form(None),
    date_from: Optional[str] = Form(None),
    date_to: Optional[str] = Form(None),
    generate_report: Optional[bool] = Form(False),
):
    gl_bytes = await gl_file.read()
    src_bytes = await src_file.read()
    map_bytes = await mapping_file.read()

    try:
        params = ReconParams.from_strings(
            amount_tol=amount_tol, date_tol=date_tol, date_from=date_from, date_to=date_to
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid parameters: {exc}")

    result = run_reconciliation(
        gl_bytes=gl_bytes,
        gl_name=gl_file.filename,
        src_bytes=src_bytes,
        src_name=src_file.filename,
        map_bytes=map_bytes,
        map_name=mapping_file.filename,
        params=params,
    )

    report_markdown = None
    report_error = None
    if generate_report:
        try:
            report_markdown = build_llm_report(result)
        except LLMUnavailable as exc:
            report_error = str(exc)

    token = uuid.uuid4().hex
    output_path = REPORTS_DIR / f"gl_recon_report_{token}.xlsx"
    build_excel_report(result, output_path)
    report_registry[token] = output_path

    return {
        "report_name": output_path.name,
        "download_url": f"/api/download/{token}",
        "summary": result.jsonable_summary(),
        "report_markdown": report_markdown,
        "report_error": report_error,
    }


@app.get("/api/download/{token}")
async def download_report(token: str):
    path = report_registry.get(token)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="Report not found or expired")
    return FileResponse(path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename=path.name)
