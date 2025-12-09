from pathlib import Path
from typing import List

import pandas as pd
from xlsxwriter.utility import xl_col_to_name

from .recon import STATUS_ORDER, ReconResult


def _auto_fit_columns(worksheet, dataframe: pd.DataFrame, min_width: int = 10, max_width: int = 40):
    for idx, col in enumerate(dataframe.columns):
        series = dataframe[col].astype(str).replace("nan", "")
        max_len = max([len(str(col))] + [len(str(x)) for x in series.tolist()])
        width = max(min_width, min(max_len + 2, max_width))
        worksheet.set_column(idx, idx, width)


def _write_table_sheet(
    writer: pd.ExcelWriter,
    sheet_name: str,
    df: pd.DataFrame,
    table_name: str,
    header_format,
    freeze: bool = True,
):
    df.to_excel(writer, sheet_name=sheet_name, startrow=1, header=False, index=False)
    ws = writer.sheets[sheet_name]
    ws.write_row(0, 0, df.columns, header_format)
    end_row = max(1, len(df))
    ws.add_table(
        0,
        0,
        end_row,
        len(df.columns) - 1,
        {"name": table_name, "columns": [{"header": c} for c in df.columns]},
    )
    if freeze:
        ws.freeze_panes(1, 0)
    _auto_fit_columns(ws, df)
    return ws


def build_excel_report(result: ReconResult, output_path: Path) -> Path:
    output_path = Path(output_path)
    status_colors = {
        "EXACT_MATCH": "#d9ead3",
        "NEAR_MATCH": "#fff2cc",
        "MAPPING_ISSUE": "#f9cb9c",
        "MISMATCH": "#f4cccc",
        "UNMATCHED_A": "#f4cccc",
        "UNMATCHED_B": "#f4cccc",
    }

    with pd.ExcelWriter(output_path, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        workbook = writer.book
        header_format = workbook.add_format(
            {"bold": True, "bg_color": "#1f2937", "font_color": "white", "border": 1}
        )
        int_format = workbook.add_format({"num_format": "#,##0"})
        money_format = workbook.add_format({"num_format": "#,##0.00"})

        # Model sheet (drives dashboard filters)
        model_ws = _write_table_sheet(writer, "Model", result.model, "ModelTable", header_format)

        # Status counts
        _write_table_sheet(writer, "StatusCounts", result.status_counts, "StatusCounts", header_format)

        # Overview sheet
        overview_df = pd.DataFrame(
            [
                ["Total rows", result.overview["total_rows"]],
                ["Exact matches", result.overview["exact_matches"]],
                ["Near matches", result.overview["near_matches"]],
                ["Mapping issues", result.overview["mapping_issues"]],
                ["Mismatches", result.overview["mismatches"]],
                ["Unmatched GL (A)", result.overview["unmatched_a"]],
                ["Unmatched Source (B)", result.overview["unmatched_b"]],
                ["Amount tolerance", result.overview["amount_tol"]],
                ["Date tolerance (days)", result.overview["date_tol"]],
            ],
            columns=["Metric", "Value"],
        )
        overview_ws = _write_table_sheet(writer, "Overview", overview_df, "Overview", header_format)
        overview_ws.set_column(1, 1, 18)
        overview_ws.set_column(0, 0, 26)

        # Matches (full model view)
        matches_ws = _write_table_sheet(writer, "Matches", result.model, "MatchesTable", header_format)

        # Conditional formatting by status
        status_col_letter = "A"
        row_count = len(result.model) + 1
        for status, color in status_colors.items():
            matches_ws.conditional_format(
                f"{status_col_letter}2:{status_col_letter}{row_count}",
                {"type": "text", "criteria": "containing", "value": status, "format": workbook.add_format({"bg_color": color})},
            )

        # Unmatched sheets
        unmatched_a_ws = _write_table_sheet(writer, "Unmatched_A", result.unmatched_a, "UnmatchedA", header_format)
        unmatched_b_ws = _write_table_sheet(writer, "Unmatched_B", result.unmatched_b, "UnmatchedB", header_format)

        # Mismatched amount
        _write_table_sheet(writer, "Mismatched_Amount", result.mismatched_amount, "MismatchedAmt", header_format)

        # Mapping table
        _write_table_sheet(writer, "Mapping", result.mapping, "MappingTable", header_format)

        # Exceptions
        _write_table_sheet(writer, "ExceptionsByGL", result.exceptions_by_gl, "ExceptionsByGL", header_format)
        _write_table_sheet(writer, "ExceptionsBySrc", result.exceptions_by_src, "ExceptionsBySrc", header_format)

        # Missing map tabs
        _write_table_sheet(writer, "Missing_Map", result.missing_map, "MissingMap", header_format)
        _write_table_sheet(writer, "MissingMap_BySource", result.missing_map_by_source, "MissingMapBySrc", header_format)

        # Lists for validation
        gl_list = ["All"] + sorted(result.model["gl_account"].dropna().astype(str).unique().tolist())
        src_list = ["All"] + sorted(result.model["source_account"].dropna().astype(str).unique().tolist())
        max_len = max(len(gl_list), len(src_list), len(STATUS_ORDER))
        pad = lambda arr: arr + [""] * (max_len - len(arr))
        lists_df = pd.DataFrame(
            {
                "gl_account": pad(gl_list),
                "source_account": pad(src_list),
                "status": pad(STATUS_ORDER),
            }
        )
        lists_ws = _write_table_sheet(writer, "Lists", lists_df, "ListsTable", header_format)
        lists_ws.set_column(0, 2, 20)

        # Dashboard sheet
        dash_ws = workbook.add_worksheet("Dashboard")
        dash_ws.set_column("A:A", 26)
        dash_ws.set_column("B:E", 20)
        dash_ws.write("A1", "GL Reconciliation Dashboard", workbook.add_format({"bold": True, "font_size": 14}))

        dash_ws.write("A2", "GL account filter")
        dash_ws.write("A3", "Source account filter")
        dash_ws.write("A4", "Date from (posting)")
        dash_ws.write("A5", "Date to (posting)")
        gl_range = f"=Lists!$A$2:$A${len(gl_list)+1}"
        src_range = f"=Lists!$B$2:$B${len(src_list)+1}"
        dash_ws.data_validation("B2", {"validate": "list", "source": gl_range})
        dash_ws.data_validation("B3", {"validate": "list", "source": src_range})
        dash_ws.write("B2", "All")
        dash_ws.write("B3", "All")

        # Status counts block (rows 8 onward) using StatusCounts table
        dash_ws.write("A8", "Status")
        dash_ws.write("B8", "Count")
        for idx, row in result.status_counts.iterrows():
            excel_row = 9 + idx  # 1-based
            dash_ws.write(excel_row - 1, 0, row["status"])
            dash_ws.write_number(excel_row - 1, 1, row["count"], int_format)

        # Exceptions by GL (top 8)
        dash_ws.write("D8", "GL Account")
        dash_ws.write("E8", "Exceptions")
        for idx, row in result.exceptions_by_gl.head(8).reset_index(drop=True).iterrows():
            excel_row = 9 + idx
            dash_ws.write(excel_row - 1, 3, row["gl_bucket"])
            dash_ws.write_number(excel_row - 1, 4, row["count"], int_format)

        # Charts
        status_chart = workbook.add_chart({"type": "column"})
        last_status_row = 8 + len(result.status_counts)
        status_chart.add_series(
            {
                "name": "Counts by Status",
                "categories": f"=Dashboard!$A$9:$A${last_status_row}",
                "values": f"=Dashboard!$B$9:$B${last_status_row}",
                "fill": {"color": "#2563eb"},
            }
        )
        status_chart.set_title({"name": "Counts by Status"})
        status_chart.set_legend({"none": True})
        status_chart.set_size({"width": 520, "height": 320})
        dash_ws.insert_chart("G2", status_chart)

        exc_chart = workbook.add_chart({"type": "bar"})
        gl_rows = min(len(result.exceptions_by_gl), 8)
        if gl_rows > 0:
            exc_chart.add_series(
                {
                    "name": "Exceptions by GL",
                    "categories": f"=Dashboard!$D$9:$D${8 + gl_rows}",
                    "values": f"=Dashboard!$E$9:$E${8 + gl_rows}",
                    "fill": {"color": "#ef4444"},
                }
            )
            exc_chart.set_title({"name": "Exceptions by GL"})
            exc_chart.set_legend({"none": True})
            exc_chart.set_size({"width": 520, "height": 320})
            dash_ws.insert_chart("G18", exc_chart)

        # Notes
        dash_ws.write("A15", "Snapshot charts pull from StatusCounts and Exceptions sheets.")

        # Shade unmatched sheets red for visibility
        for ws in [unmatched_a_ws, unmatched_b_ws]:
            if ws.dim_rowmax is None:
                continue
            last_row = (ws.dim_rowmax or 0) + 1
            if last_row < 2:
                continue
            last_col = ws.dim_colmax or 0
            end_col_letter = xl_col_to_name(last_col)
            ws.conditional_format(
                f"A2:{end_col_letter}{last_row}",
                {
                    "type": "no_blanks",
                    "format": workbook.add_format({"bg_color": status_colors["MISMATCH"]}),
                },
            )

        # Force Dashboard to be first tab
        workbook.worksheets_objs.insert(0, workbook.worksheets_objs.pop())

    return output_path
