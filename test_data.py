from databricks import sql
from dotenv import load_dotenv
import os

load_dotenv()  # reads your .env file

conn = sql.connect(
    server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
    http_path=os.getenv("DATABRICKS_HTTP_PATH"),
    access_token=os.getenv("DATABRICKS_ACCESS_TOKEN")
)

catalog = os.getenv("DATABRICKS_CATALOG")
schema = os.getenv("DATABRICKS_SCHEMA")

cursor = conn.cursor()

# See all tables
cursor.execute(f"SHOW TABLES IN {catalog}.{schema}")
tables = cursor.fetchall()

print("=== TABLES FOUND ===")
for table in tables:
    print(table)
tables_list = ["dim_aw_master", "dim_customer_master", "dim_product_master", "fact_invoice"]

for table in tables_list:
    print(f"\n=== {table} ===")
    cursor.execute(f"SELECT * FROM {catalog}.{schema}.{table} LIMIT 3")
    rows = cursor.fetchall()
    for row in rows:
        print(row)
        
conn.close()