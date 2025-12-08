import os
from typing import List, Optional

from openai import OpenAI, OpenAIError

from .recon import ReconResult


class LLMUnavailable(Exception):
    pass


def _fmt_table(rows: List[List[str]], headers: List[str]) -> str:
    if not rows:
        return "None"
    col_widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    def fmt_row(r): return " | ".join(str(r[i]).ljust(col_widths[i]) for i in range(len(headers)))
    return "\n".join([fmt_row(headers), "-|-".join("-" * w for w in col_widths), *[fmt_row(r) for r in rows]])


def build_llm_report(result: ReconResult, model: str = "gpt-4o-mini") -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise LLMUnavailable("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)

    status_counts = result.status_counts.to_dict(orient="records")
    exceptions_by_gl = result.exceptions_by_gl.head(10).to_dict(orient="records")
    exceptions_by_src = result.exceptions_by_src.head(10).to_dict(orient="records")
    mismatches = result.mismatched_amount.head(10)
    missing_map = result.missing_map_by_source.head(10).to_dict(orient="records")

    status_table = _fmt_table(
        [[s["status"], s["count"]] for s in status_counts],
        ["Status", "Count"],
    )
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
    mismatch_table = _fmt_table(mismatch_rows, ["Doc/Ref", "Amount Δ", "Days Δ", "GL", "Source"])
    missing_table = _fmt_table(
        [[m.get("source_account", ""), m.get("count", 0)] for m in missing_map],
        ["Source Account", "Missing map count"],
    )

    overview = result.overview
    prompt = f"""
You are a finance controller. Write a concise client-facing reconciliation summary based on the provided numbers.

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

Write in Markdown with sections: Overview, Highlights (bulleted), Exceptions (bulleted), Missing Mapping, Next Steps (3-5 bullets). Keep it under 220 words and avoid restating the raw tables verbatim—reference patterns and magnitudes instead.
"""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You write short, precise finance reconciliation summaries. Use Markdown headings and bullets."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        return completion.choices[0].message.content.strip()
    except OpenAIError as exc:  # noqa: BLE001
        raise LLMUnavailable(f"LLM call failed: {exc}")
