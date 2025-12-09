import io
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .config import ReconParams


GL_COLUMNS = [
    "gl_account",
    "gl_account_name",
    "posting_date",
    "doc_id",
    "amount",
    "currency",
    "entity",
    "cost_center",
    "counterparty",
    "memo",
]

SRC_COLUMNS = [
    "source_account",
    "source_account_name",
    "txn_date",
    "ref_id",
    "amount",
    "currency",
    "entity",
    "cost_center",
    "counterparty",
    "description",
]

MAP_COLUMNS = ["source_account", "gl_account"]

STATUS_ORDER = [
    "EXACT_MATCH",
    "NEAR_MATCH",
    "MAPPING_ISSUE",
    "MISMATCH",
    "UNMATCHED_A",
    "UNMATCHED_B",
]


@dataclass
class ReconResult:
    model: pd.DataFrame
    status_counts: pd.DataFrame
    overview: Dict[str, float]
    issues: List[str]
    data_quality_flags: List[str]
    unmatched_a: pd.DataFrame
    unmatched_b: pd.DataFrame
    mismatched_amount: pd.DataFrame
    mapping: pd.DataFrame
    exceptions_by_gl: pd.DataFrame
    exceptions_by_src: pd.DataFrame
    missing_map: pd.DataFrame
    missing_map_by_source: pd.DataFrame
    params: ReconParams

    def jsonable_summary(self, sample_rows: int = 25) -> Dict:
        def frame_to_records(df: pd.DataFrame, n: int) -> List[Dict]:
            if df.empty:
                return []
            cleaned = df.head(n).replace({np.nan: None, pd.NA: None})
            return cleaned.to_dict(orient="records")

        def clean_overview(ov: Dict[str, float]) -> Dict[str, Optional[float]]:
            return {k: (v if v is not None and not pd.isna(v) else 0) for k, v in ov.items()}

        status_counts_records = (
            self.status_counts.where(pd.notna(self.status_counts), None).to_dict(orient="records")
            if not self.status_counts.empty
            else []
        )
        mismatch_by_gl = (
            self.model[self.model["status"].isin(["MISMATCH", "UNMATCHED_A", "UNMATCHED_B", "MAPPING_ISSUE"])]
            .groupby("gl_account")
            .size()
            .reset_index(name="count")
            .replace({np.nan: None, pd.NA: None})
            .to_dict(orient="records")
        )
        mismatch_by_src = (
            self.model[self.model["status"].isin(["MISMATCH", "UNMATCHED_A", "UNMATCHED_B", "MAPPING_ISSUE"])]
            .groupby("source_account")
            .size()
            .reset_index(name="count")
            .replace({np.nan: None, pd.NA: None})
            .to_dict(orient="records")
        )
        mismatch_by_counterparty = (
            self.model[self.model["status"].isin(["MISMATCH", "UNMATCHED_A", "UNMATCHED_B", "MAPPING_ISSUE"])]
            .groupby("counterparty_A")
            .size()
            .reset_index(name="count")
            .replace({np.nan: None, pd.NA: None})
            .to_dict(orient="records")
        )
        mismatch_by_entity = (
            self.model[self.model["status"].isin(["MISMATCH", "UNMATCHED_A", "UNMATCHED_B", "MAPPING_ISSUE"])]
            .groupby("entity_A")
            .size()
            .reset_index(name="count")
            .replace({np.nan: None, pd.NA: None})
            .to_dict(orient="records")
        )
        mismatch_by_cost_center = (
            self.model[self.model["status"].isin(["MISMATCH", "UNMATCHED_A", "UNMATCHED_B", "MAPPING_ISSUE"])]
            .groupby("cost_center_A")
            .size()
            .reset_index(name="count")
            .replace({np.nan: None, pd.NA: None})
            .to_dict(orient="records")
        )
        ts_df = self.model.copy()
        ts_df["date_for_ts"] = ts_df["posting_date"].combine_first(ts_df["txn_date"])
        ts_df["month"] = ts_df["date_for_ts"].dt.to_period("M").astype(str)
        ts_df["year"] = ts_df["date_for_ts"].dt.year
        ts_df["week"] = ts_df["date_for_ts"].dt.to_period("W-MON").astype(str)
        status_ts_month = (
            ts_df.dropna(subset=["month"])
            .groupby(["month", "status"])
            .size()
            .reset_index(name="count")
            .sort_values("month")
            .replace({np.nan: None, pd.NA: None})
            .to_dict(orient="records")
        )
        status_ts_week = (
            ts_df.dropna(subset=["week"])
            .groupby(["week", "status"])
            .size()
            .reset_index(name="count")
            .sort_values("week")
            .replace({np.nan: None, pd.NA: None})
            .to_dict(orient="records")
        )
        status_ts_year = (
            ts_df.dropna(subset=["year"])
            .groupby(["year", "status"])
            .size()
            .reset_index(name="count")
            .sort_values("year")
            .replace({np.nan: None, pd.NA: None})
            .to_dict(orient="records")
        )
        return {
            "overview": clean_overview(self.overview),
            "status_counts": status_counts_records,
            "issues": self.issues,
            "data_quality_flags": self.data_quality_flags,
            "sample": frame_to_records(self.model, sample_rows),
            "unmatched_a": len(self.unmatched_a),
            "unmatched_b": len(self.unmatched_b),
            "mismatched_amount": len(self.mismatched_amount),
            "mismatched_amount_sample": frame_to_records(self.mismatched_amount, 15),
            "exceptions_by_gl": frame_to_records(self.exceptions_by_gl, 15),
            "exceptions_by_src": frame_to_records(self.exceptions_by_src, 15),
            "missing_map_by_source": frame_to_records(self.missing_map_by_source, 15),
            "mismatch_by_gl": mismatch_by_gl,
            "mismatch_by_src": mismatch_by_src,
            "mismatch_by_counterparty": mismatch_by_counterparty,
            "mismatch_by_entity": mismatch_by_entity,
            "mismatch_by_cost_center": mismatch_by_cost_center,
            "status_time_series_month": status_ts_month,
            "status_time_series_week": status_ts_week,
            "status_time_series_year": status_ts_year,
            "params": {
                "amount_tol": self.params.amount_tol,
                "date_tol": self.params.date_tol,
                "date_from": self.params.date_from.isoformat() if self.params.date_from else None,
                "date_to": self.params.date_to.isoformat() if self.params.date_to else None,
            },
        }


def read_input_file(
    file_bytes: bytes, filename: str, expected_columns: List[str], date_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    issues: List[str] = []
    ext = Path(filename).suffix.lower()
    buffer = io.BytesIO(file_bytes)

    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(buffer)
    else:
        df = pd.read_csv(buffer)

    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        issues.append(f"{filename}: missing columns {missing}")
    extra = [c for c in df.columns if c not in expected_columns]
    if extra:
        issues.append(f"{filename}: extra columns {extra}")

    for col in missing:
        df[col] = pd.NA

    if date_columns:
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    return df, issues


def normalize_value(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_status_counts(status_series: pd.Series) -> pd.DataFrame:
    counter = Counter(status_series.fillna("MISMATCH"))
    data = [{"status": status, "count": int(counter.get(status, 0))} for status in STATUS_ORDER]
    return pd.DataFrame(data)


def run_reconciliation(
    gl_bytes: bytes,
    gl_name: str,
    src_bytes: bytes,
    src_name: str,
    map_bytes: bytes,
    map_name: str,
    params: ReconParams,
) -> ReconResult:
    issues: List[str] = []
    gl_df, gl_issues = read_input_file(gl_bytes, gl_name, GL_COLUMNS, date_columns=["posting_date"])
    src_df, src_issues = read_input_file(src_bytes, src_name, SRC_COLUMNS, date_columns=["txn_date"])
    mapping_df, map_issues = read_input_file(map_bytes, map_name, MAP_COLUMNS)

    issues.extend(gl_issues + src_issues + map_issues)

    for df, col in [(gl_df, "amount"), (src_df, "amount")]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Apply optional date filters before matching
    if params.date_from:
        gl_df = gl_df[gl_df["posting_date"] >= params.date_from]
        src_df = src_df[src_df["txn_date"] >= params.date_from]
    if params.date_to:
        gl_df = gl_df[gl_df["posting_date"] <= params.date_to]
        src_df = src_df[src_df["txn_date"] <= params.date_to]

    # Mapping cleanup and lookup
    mapping_df["source_account"] = mapping_df["source_account"].apply(lambda x: x if normalize_value(x) else pd.NA)
    mapping_df["gl_account"] = mapping_df["gl_account"].apply(lambda x: x if normalize_value(x) else pd.NA)
    mapping_lookup = (
        mapping_df.dropna(subset=["source_account"])
        .set_index("source_account")["gl_account"]
        .to_dict()
    )
    src_df["mapped_gl_account"] = src_df["source_account"].map(mapping_lookup)
    src_df["mapped_gl_account"] = src_df["mapped_gl_account"].apply(lambda x: x if normalize_value(x) else pd.NA)

    merged = gl_df.merge(
        src_df,
        left_on="doc_id",
        right_on="ref_id",
        how="outer",
        suffixes=("_A", "_B"),
    )

    merged["amount_A"] = pd.to_numeric(merged.get("amount_A"), errors="coerce")
    merged["amount_B"] = pd.to_numeric(merged.get("amount_B"), errors="coerce")

    merged["amount_diff"] = merged["amount_A"].fillna(0) - merged["amount_B"].fillna(0)
    merged["days_diff"] = (merged["posting_date"] - merged["txn_date"]).dt.days

    a_exists = merged["doc_id"].notna()
    b_exists = merged["ref_id"].notna()

    norm_gl = merged["gl_account"].apply(normalize_value)
    norm_mapped = merged["mapped_gl_account"].apply(normalize_value)
    mapping_ok_series = (b_exists & (norm_gl == norm_mapped)).astype(bool)
    merged["mapping_ok"] = mapping_ok_series

    within_amount = merged["amount_diff"].abs() <= params.amount_tol
    within_date = merged["days_diff"].abs() <= params.date_tol
    both_exist = a_exists & b_exists

    merged["status"] = "MISMATCH"
    merged.loc[~a_exists & b_exists, "status"] = "UNMATCHED_B"
    merged.loc[a_exists & ~b_exists, "status"] = "UNMATCHED_A"
    merged.loc[both_exist & mapping_ok_series & (merged["amount_diff"] == 0) & (merged["days_diff"] == 0), "status"] = (
        "EXACT_MATCH"
    )
    merged.loc[both_exist & mapping_ok_series & within_amount & within_date & (merged["status"] != "EXACT_MATCH"), "status"] = (
        "NEAR_MATCH"
    )
    merged.loc[both_exist & (~mapping_ok_series) & within_amount & within_date, "status"] = "MAPPING_ISSUE"

    # Data quality check
    dq_flags: List[str] = []

    status_counts = build_status_counts(merged["status"])

    overview = {
        "total_rows": int(len(merged)),
        "exact_matches": int(status_counts[status_counts["status"] == "EXACT_MATCH"]["count"].iloc[0]),
        "near_matches": int(status_counts[status_counts["status"] == "NEAR_MATCH"]["count"].iloc[0]),
        "mapping_issues": int(status_counts[status_counts["status"] == "MAPPING_ISSUE"]["count"].iloc[0]),
        "mismatches": int(status_counts[status_counts["status"] == "MISMATCH"]["count"].iloc[0]),
        "unmatched_a": int(status_counts[status_counts["status"] == "UNMATCHED_A"]["count"].iloc[0]),
        "unmatched_b": int(status_counts[status_counts["status"] == "UNMATCHED_B"]["count"].iloc[0]),
        "amount_tol": params.amount_tol,
        "date_tol": params.date_tol,
    }

    mismatched_amount = merged[
        merged["status"].isin(["MISMATCH", "MAPPING_ISSUE"])
        & merged["amount_diff"].abs().gt(params.amount_tol)
    ]
    unmatched_a = merged[merged["status"] == "UNMATCHED_A"]
    unmatched_b = merged[merged["status"] == "UNMATCHED_B"]

    # Exceptions pivots
    def gl_label(row) -> str:
        if not pd.isna(row.get("gl_account")):
            return str(row["gl_account"])
        if not pd.isna(row.get("mapped_gl_account")):
            return str(row["mapped_gl_account"])
        return "Unmapped"

    exc_gl_df = merged[merged["status"].isin(["MISMATCH", "UNMATCHED_A"])].copy()
    exc_gl_df["gl_bucket"] = exc_gl_df.apply(gl_label, axis=1)
    exceptions_by_gl = (
        exc_gl_df.groupby("gl_bucket")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    exc_src_df = merged[merged["status"].isin(["MISMATCH", "UNMATCHED_B"])].copy()
    exc_src_df["source_bucket"] = exc_src_df["source_account"].fillna("Unknown").astype(str)
    exceptions_by_src = (
        exc_src_df.groupby("source_bucket")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    missing_map = src_df[src_df["mapped_gl_account"].isna()].copy()
    missing_map = missing_map[
        [
            "source_account",
            "source_account_name",
            "ref_id",
            "txn_date",
            "amount",
            "currency",
            "entity",
            "cost_center",
            "counterparty",
            "description",
        ]
    ]
    missing_map_by_source = (
        missing_map.groupby("source_account")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    # Ensure consistent column order for model
    model_columns = [
        "status",
        "mapping_ok",
        "amount_diff",
        "days_diff",
        "gl_account",
        "gl_account_name",
        "posting_date",
        "doc_id",
        "amount_A",
        "currency_A",
        "entity_A",
        "cost_center_A",
        "counterparty_A",
        "memo",
        "source_account",
        "source_account_name",
        "txn_date",
        "ref_id",
        "amount_B",
        "currency_B",
        "entity_B",
        "cost_center_B",
        "counterparty_B",
        "description",
        "mapped_gl_account",
    ]

    for col in model_columns:
        if col not in merged.columns:
            merged[col] = pd.NA

    merged = merged[model_columns]

    return ReconResult(
        model=merged,
        status_counts=status_counts,
        overview=overview,
        issues=issues,
        data_quality_flags=dq_flags,
        unmatched_a=unmatched_a,
        unmatched_b=unmatched_b,
        mismatched_amount=mismatched_amount,
        mapping=mapping_df,
        exceptions_by_gl=exceptions_by_gl,
        exceptions_by_src=exceptions_by_src,
        missing_map=missing_map,
        missing_map_by_source=missing_map_by_source,
        params=params,
    )
