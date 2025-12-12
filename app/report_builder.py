import os
from typing import Dict, List

import httpx
import pandas as pd
from openai import OpenAI, OpenAIError

from .recon import ReconResult


class LLMUnavailable(Exception):
    """Raised when an LLM call cannot be fulfilled (missing key, quota, bad response)."""


def _fmt_table(rows: List[List[str]], headers: List[str]) -> str:
    if not rows:
        return "None"
    col_widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]

    def fmt_row(r):  # noqa: ANN001
        return " | ".join(str(r[i]).ljust(col_widths[i]) for i in range(len(headers)))

    return "\n".join([fmt_row(headers), "-|-".join("-" * w for w in col_widths), *[fmt_row(r) for r in rows]])


def _summary_payload(result: ReconResult) -> Dict:
    """Structured JSON for downstream LLMs (OpenAI/Vellum)."""
    summary = result.jsonable_summary(sample_rows=25)
    summary["exceptions_by_gl"] = result.exceptions_by_gl.head(10).to_dict(orient="records")
    summary["exceptions_by_src"] = result.exceptions_by_src.head(10).to_dict(orient="records")
    summary["mismatched_amount_sample"] = (
        result.mismatched_amount.head(10).replace({pd.NA: None}).to_dict(orient="records")
        if not result.mismatched_amount.empty
        else []
    )
    summary["missing_map_by_source"] = (
        result.missing_map_by_source.head(10).replace({pd.NA: None}).to_dict(orient="records")
        if not result.missing_map_by_source.empty
        else []
    )
    return _sanitize(summary)

def build_analysis_payload(result: ReconResult) -> Dict:
    """Public helper for generating the sanitized analysis JSON."""
    return _summary_payload(result)


def _sanitize(obj):
    """Recursively make data JSON-serializable (timestamps -> iso, NaN -> None)."""
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, float) and pd.isna(obj):
        return None
    if obj is pd.NA:
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def build_report(result: ReconResult) -> str:
    """Entry point. Uses Vellum if LLM_PROVIDER=VELLUM, else OpenAI."""
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider == "vellum":
        return build_vellum_report(result)
    return build_openai_report(result)


def build_vellum_report(result: ReconResult) -> str:
    api_key = os.environ.get("VELLUM_API_KEY")
    deployment_id = os.environ.get("VELLUM_DEPLOYMENT_ID")
    workflow_deployment_id = os.environ.get("VELLUM_WORKFLOW_DEPLOYMENT_ID")
    base_url = os.environ.get("VELLUM_BASE_URL", "https://api.vellum.ai")
    if not api_key:
        raise LLMUnavailable("VELLUM_API_KEY is not set")
    if not (deployment_id or workflow_deployment_id):
        raise LLMUnavailable("VELLUM_DEPLOYMENT_ID or VELLUM_WORKFLOW_DEPLOYMENT_ID is not set")

    payload = {"inputs": {"analysis_json": _summary_payload(result)}}
    if workflow_deployment_id:
        url = f"{base_url.rstrip('/')}/v1/workflow-deployments/{workflow_deployment_id}/execute"
    else:
        url = f"{base_url.rstrip('/')}/v1/deployments/{deployment_id}/execute"
    try:
        resp = httpx.post(url, headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=30)
    except httpx.HTTPError as exc:  # noqa: BLE001
        raise LLMUnavailable(f"Vellum call failed: {exc}")

    if resp.status_code >= 300:
        raise LLMUnavailable(f"Vellum call failed ({resp.status_code}): {resp.text}")

    data = resp.json()
    text = None
    if isinstance(data, dict):
        outputs = data.get("outputs") or data.get("data")
        if isinstance(outputs, list) and outputs:
            text = outputs[0].get("value") or outputs[0].get("text")
        elif isinstance(outputs, dict):
            inner = outputs.get("outputs")
            if isinstance(inner, list) and inner:
                text = inner[0].get("value") or inner[0].get("text")
    return text or resp.text


def build_openai_report(result: ReconResult, model: str = "gpt-4o-mini") -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise LLMUnavailable("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key, http_client=httpx.Client())

    status_counts = result.status_counts.to_dict(orient="records")
    exceptions_by_gl = result.exceptions_by_gl.head(10).to_dict(orient="records")
    exceptions_by_src = result.exceptions_by_src.head(10).to_dict(orient="records")
    mismatches = result.mismatched_amount.head(10)
    missing_map = result.missing_map_by_source.head(10).to_dict(orient="records")

    status_table = _fmt_table([[s["status"], s["count"]] for s in status_counts], ["Status", "Count"])
    gl_table = _fmt_table([[g.get("gl_bucket", ""), g.get("count", 0)] for g in exceptions_by_gl], ["GL", "Exceptions"])
    src_table = _fmt_table([[s.get("source_bucket", ""), s.get("count", 0)] for s in exceptions_by_src], ["Source", "Exceptions"])
    mismatch_rows = []
    for _, row in mismatches.iterrows():
        mismatch_rows.append(
            [
                row.get("doc_id") or row.get("ref_id") or "",
                f"{row.get('amount_diff', 0):,.2f}",
                row.get("days_diff", ""),
                row.get("gl_account", ""),
                row.get("source_account", ""),
            ]
        )
    mismatch_table = _fmt_table(
        mismatch_rows,
        ["Doc/Ref", "Amount diff", "Days diff", "GL", "Source"],
    )
    missing_table = _fmt_table(
        [[m.get("source_account", ""), m.get("count", 0)] for m in missing_map],
        ["Source Account", "Missing map count"],
    )

    overview = result.overview
    prompt = f"""
You are a finance controller. Write a detailed client-facing reconciliation narrative based on the provided numbers. Target roughly 500–750 words so it feels substantive.

Key metrics:
- Total rows: {overview.get('total_rows')}
- Exact matches: {overview.get('exact_matches')}
- Near matches: {overview.get('near_matches')}
- Mismatches: {overview.get('mismatches')}
- Unmatched A: {overview.get('unmatched_a')}
- Unmatched B: {overview.get('unmatched_b')}
- Amount tolerance: {overview.get('amount_tol')}
- Date tolerance (days): {overview.get('date_tol')}

Status counts:
{status_table}

Exceptions by GL (top 10):
{gl_table}

Exceptions by Source (top 10):
{src_table}

Mismatched pairs (top 10 by amount diff):
{mismatch_table}

Missing mappings by source (top 10):
{missing_table}

Write in Markdown with clear headings and short bullet lists. Required sections:
- Overview: 2–3 sentences with overall health and tolerances.
- Mix & trends: bullets on status mix, notable shifts or concentrations (reference counts and magnitude, not whole tables).
- Key drivers: bullets for top GL accounts and Source accounts driving exceptions, and the dominant mismatch reasons (amount vs date vs mapping).
- Exceptions: bullets summarizing biggest mismatches, including at least one doc/ref example with amount and days difference if available.
- Mapping & missing: bullets on mapping issues and missing mappings by source (counts and any clear pattern).
- Next steps: 4–6 concise action bullets tailored to the data quality issues observed.

Keep it factual; do not paste the raw tables; summarize patterns and size. Avoid line-leading hashes inside bullets; use proper headings and bullets.
"""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You write detailed but concise finance reconciliation summaries. Use Markdown headings and bullets; avoid code blocks."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1400,
        )
        return completion.choices[0].message.content.strip()
    except OpenAIError as exc:  # noqa: BLE001
        raise LLMUnavailable(f"LLM call failed: {exc}")
