from databricks import sql
from dotenv import load_dotenv
import os


TABLES = [
    "llm_test.llm.invoice_template",
    "llm_test.llm.target_template",
]

REQUIRED_MONTHS = ["2025-03", "2026-02", "2026-03"]


def _quote_ident(name):
    return "`" + str(name or "").replace("`", "``") + "`"


def _quote_table_fqn(table_fqn):
    parts = [p.strip() for p in str(table_fqn or "").split(".") if p.strip()]
    if not parts:
        raise ValueError(f"Invalid table name: {table_fqn}")
    return ".".join(_quote_ident(p) for p in parts)


def _print_table_preview(cursor, table_name):
    quoted_table = _quote_table_fqn(table_name)
    query = f"SELECT * FROM {quoted_table} LIMIT 5"

    print("=" * 100)
    print(f"TABLE: {table_name}")
    print(f"QUERY: {query}")
    print("=" * 100)

    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [c[0] for c in (cursor.description or [])]

    print(f"Total columns: {len(columns)}")
    print("Columns:")
    for col in columns:
        print(f"- {col}")

    print(f"\nTop {len(rows)} row(s):")
    if not rows:
        print("(No rows returned)")
        print()
        return

    for idx, row in enumerate(rows, start=1):
        print(f"\n--- Row {idx} ---")
        for col_name, value in zip(columns, row):
            print(f"{col_name}: {value}")
    print()

    _print_month_summary(cursor, table_name, columns)


def _pick_month_column(columns):
    if not columns:
        return None
    col_map = {str(c).strip().lower(): str(c).strip() for c in columns}
    for cand in ["date", "monthly", "yyyy_mm", "month"]:
        hit = col_map.get(cand)
        if hit:
            return hit
    for c in columns:
        n = str(c).strip().lower()
        if "date" in n or "month" in n:
            return str(c).strip()
    return None


def _print_month_summary(cursor, table_name, columns):
    month_col = _pick_month_column(columns)
    if not month_col:
        print("Month summary: skipped (no date/month column detected)")
        print()
        return

    quoted_table = _quote_table_fqn(table_name)
    quoted_col = _quote_ident(month_col)
    month_sql = (
        f"SELECT SUBSTR(CAST({quoted_col} AS STRING), 1, 7) AS month_key, COUNT(*) AS row_count "
        f"FROM {quoted_table} "
        f"WHERE {quoted_col} IS NOT NULL "
        f"AND TRIM(CAST({quoted_col} AS STRING)) <> '' "
        f"GROUP BY 1 ORDER BY 1"
    )

    print("-" * 100)
    print(f"Month summary column: {month_col}")
    print(f"MONTH QUERY: {month_sql}")
    cursor.execute(month_sql)
    month_rows = cursor.fetchall()

    if not month_rows:
        print("No month data found.")
        print()
        return

    print("Available months (YYYY-MM):")
    month_counts = {}
    for month_key, row_count in month_rows:
        mk = str(month_key or "").strip()
        if not mk:
            continue
        month_counts[mk] = int(row_count or 0)
        print(f"- {mk}: {int(row_count or 0):,} rows")

    print("\nRequired month check:")
    for required in REQUIRED_MONTHS:
        count = month_counts.get(required, 0)
        status = "PRESENT" if count > 0 else "MISSING"
        print(f"- {required}: {status} ({count:,} rows)")
    print()


def main():
    load_dotenv()

    host = (os.getenv("DATABRICKS_SERVER_HOSTNAME") or "").strip()
    http_path = (os.getenv("DATABRICKS_HTTP_PATH") or "").strip()
    token = (os.getenv("DATABRICKS_ACCESS_TOKEN") or "").strip()

    if not host or not http_path or not token:
        raise RuntimeError(
            "Missing Databricks credentials. "
            "Set DATABRICKS_SERVER_HOSTNAME, DATABRICKS_HTTP_PATH, DATABRICKS_ACCESS_TOKEN."
        )

    connection = sql.connect(
        server_hostname=host,
        http_path=http_path,
        access_token=token,
    )

    try:
        with connection.cursor() as cursor:
            for table in TABLES:
                _print_table_preview(cursor, table)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
