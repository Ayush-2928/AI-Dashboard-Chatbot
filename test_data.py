from databricks import sql
from dotenv import load_dotenv
import os


def _quote_ident(name):
    return "`" + str(name or "").replace("`", "``") + "`"


load_dotenv()  # reads your .env file

host = os.getenv("DATABRICKS_SERVER_HOSTNAME")
http_path = os.getenv("DATABRICKS_HTTP_PATH")
token = os.getenv("DATABRICKS_ACCESS_TOKEN")
catalog = os.getenv("DATABRICKS_CATALOG")
schema = os.getenv("DATABRICKS_SCHEMA")

if not host or not http_path or not token:
    raise RuntimeError("Missing Databricks connection env vars")
if not catalog or not schema:
    raise RuntimeError("Missing DATABRICKS_CATALOG / DATABRICKS_SCHEMA")

fact_table = f"{_quote_ident(catalog)}.{_quote_ident(schema)}.{_quote_ident('fact_invoice')}"
product_table = f"{_quote_ident(catalog)}.{_quote_ident(schema)}.{_quote_ident('dim_product_master')}"
customer_table = f"{_quote_ident(catalog)}.{_quote_ident(schema)}.{_quote_ident('dim_customer_master')}"

query = f"""
SELECT
    f.*,
    p.*,
    c.*
FROM {fact_table} f
LEFT JOIN {product_table} p
    ON TRIM(UPPER(CAST(f.{_quote_ident('Material_No')} AS STRING)))
     = TRIM(UPPER(CAST(p.{_quote_ident('ITEM_CBU_CODE')} AS STRING)))
LEFT JOIN {customer_table} c
    ON TRIM(UPPER(CAST(f.{_quote_ident('AW_CODE')} AS STRING)))
     = TRIM(UPPER(CAST(c.{_quote_ident('Sold_to_Code')} AS STRING)))
LIMIT 3
""".strip()

conn = sql.connect(
    server_hostname=host,
    http_path=http_path,
    access_token=token,
)

try:
    with conn.cursor() as cursor:
        print("=== JOIN QUERY (TOP 3) ===")
        print(query)
        print()

        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [c[0] for c in cursor.description] if cursor.description else []

        print(f"Total columns: {len(columns)}")
        print("Columns:")
        print(columns)
        print()

        print(f"Returned rows: {len(rows)}")
        for idx, row in enumerate(rows, start=1):
            print(f"\n--- Row {idx} ---")
            for col_name, value in zip(columns, row):
                print(f"{col_name}: {value}")
finally:
    conn.close()
