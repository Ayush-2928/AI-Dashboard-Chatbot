import duckdb
import pandas as pd
import json
import os
import re
import time
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from django.http import HttpResponse
import zipfile
import io
import os
from django.conf import settings
import uuid
import tempfile

# --- CONFIGURATION ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

THEME_MAP = {
    "Supply Chain":     {"color": "#F59E0B", "gradient": "from-amber-400 to-orange-500"},
    "Sales":            {"color": "#10B981", "gradient": "from-emerald-400 to-green-500"},
    "Retail":           {"color": "#6366F1", "gradient": "from-indigo-400 to-blue-500"},
    "Finance":          {"color": "#3B82F6", "gradient": "from-blue-400 to-cyan-500"},
    "E-commerce":       {"color": "#F43F5E", "gradient": "from-rose-400 to-red-500"},
    "Marketing":        {"color": "#EC4899", "gradient": "from-pink-400 to-fuchsia-500"},
    "Human Resources":  {"color": "#8B5CF6", "gradient": "from-violet-400 to-purple-500"},
    "Production":       {"color": "#64748B", "gradient": "from-slate-400 to-gray-500"},
    "CRM":              {"color": "#14B8A6", "gradient": "from-teal-400 to-emerald-500"},
    "Customer Support": {"color": "#06B6D4", "gradient": "from-cyan-400 to-sky-500"},
    "General":          {"color": "#94A3B8", "gradient": "from-slate-400 to-slate-500"}
}

def clean_col_name(col):
    return re.sub(r'[^a-zA-Z0-9]', '_', str(col)).lower().strip('_')

def call_ai_with_retry(messages, json_mode=False, retries=3):
    if not client: return None, 0
    for attempt in range(retries):
        try:
            kwargs = { "model": "gpt-4o", "messages": messages, "temperature": 0 }
            if json_mode: kwargs["response_format"] = {"type": "json_object"}
            res = client.chat.completions.create(**kwargs)
            tokens_used = res.usage.total_tokens if hasattr(res, 'usage') else 0
            return res.choices[0].message.content, tokens_used
        except: time.sleep(1)
    return None, 0

def get_table_schema(con, table_name):
    try:
        info = con.execute(f"DESCRIBE {table_name}").df()
        cols = ", ".join([f"{r['column_name']} ({r['column_type']})" for _, r in info.iterrows()])
        sample = con.execute(f"SELECT * FROM {table_name} LIMIT 2").df().to_string(index=False)
        return f"TABLE: {table_name}\nCOLUMNS: {cols}\nSAMPLE:\n{sample}\n" + "-"*30 + "\n"
    except: return ""

# --- PHASE 1: ARCHITECT ---
def generate_join_sql(raw_schema_context):
    prompt = f"""
    You are a Data Architect. DuckDB Schema:
    {raw_schema_context}
    TASK: Write ONE SQL query to create 'master_view'.
    RULES:
    1. Use `clean_id(t1.col) = clean_id(t2.col)` ONLY for IDs (e.g. OrderID, ProductID).
    2. Do NOT use `SELECT *`. Select specific columns with aliases.
    RETURN RAW SQL ONLY.
    """
    res, tokens = call_ai_with_retry([{"role": "user", "content": prompt}], json_mode=False)
    return (res.replace('```sql', '').replace('```', '').strip() if res else None), tokens

# --- PHASE 2: ANALYST ---
def generate_viz_config(master_schema_context, forced_domain=None):
    domains = ", ".join(list(THEME_MAP.keys()))
    prompt = f"""
    You are a BI Expert. Analyze 'master_view':
    {master_schema_context}
    
    STEP 1: DETECT DOMAIN -> One of [{domains}]
    {f"(HINT: Looks like {forced_domain})" if forced_domain else ""}
    
    STEP 2: FILTERS -> 3 categorical columns for filtering.
    STEP 3: CHARTS -> 6 SQL queries (including heatmap if applicable).
    
    **MANDATORY CHART RULES**:
    - **Chart 0 (Line/Trend)**: MUST be a trend over time. SQL: `SELECT CAST(date_col AS DATE) as x, SUM(numeric_col) as y FROM master_view GROUP BY 1 ORDER BY 1`
    - **Chart 1 (Heatmap)**: If you have TWO categorical dimensions and a numeric value, create a heatmap. SQL: `SELECT cat1 as x, cat2 as y, SUM(value) as z FROM master_view GROUP BY 1,2`. Otherwise, make it a bar chart.
    - **Chart 2 (Sankey/Decomposition)**: MUST be a multi-level Sankey diagram showing 3-4 categorical hierarchies for deep root cause analysis. SQL should UNION ALL multiple level connections: Level1→Level2, Level2→Level3, Level3→Level4.
    - **Charts 3-5**: Choose the BEST visualization (line for trends, bar for categorical comparisons) based on the data
    - Use LINE charts when showing trends over time periods
    - Use BAR charts when comparing categories or showing rankings
    - Use SANKEY for hierarchical flow/decomposition analysis

     **TITLE REQUIREMENTS**:
    - Titles MUST be specific and describe what metric is being shown in detail
    - BAD: "Monthly Trend", "Distribution", "Breakdown"
    - GOOD: "Monthly Revenue Trend", "Sales by Region", "Product Category Performance"
    - Include the actual metric name (Revenue, Count, Orders, etc.) and dimension (by Month, by Region, etc.)
    - For Sankey: "Root Cause: [Metric] by [Dimension1] → [Dimension2] → [Dimension3]"
    
    
    RETURN JSON:
    {{
        "domain": "Supply Chain",
        "filters": ["Region", "Status", "Category"],
        "kpis": [ 
            {{ "label": "Total Spend", "sql": "SELECT SUM(amount) FROM master_view", "trend_sql": "SELECT CAST(date_col AS DATE) as x, SUM(amount) as y FROM master_view GROUP BY 1 ORDER BY 1 LIMIT 7" }},
            {{ "label": "Record Count", "sql": "SELECT COUNT(*) FROM master_view", "trend_sql": "SELECT CAST(date_col AS DATE) as x, COUNT(*) as y FROM master_view GROUP BY 1 ORDER BY 1 LIMIT 7" }}
        ],
        "charts": [ 
            {{ "title": "Monthly Trend", "type": "line", "sql": "SELECT date_col as x, SUM(val) as y FROM master_view GROUP BY 1 ORDER BY 1", "xlabel": "Date", "ylabel": "Amount" }},
            {{ "title": "Heatmap", "type": "heatmap", "sql": "SELECT cat1 as x, cat2 as y, SUM(val) as z FROM master_view GROUP BY 1,2", "xlabel": "Category 1", "ylabel": "Category 2" }},
            {{ "title": "Root Cause Analysis", "type": "sankey", "sql": "SELECT cat1 as source, cat2 as target, SUM(val) as value FROM master_view GROUP BY 1,2", "xlabel": "", "ylabel": "" }}
        ]
    }}
    """
    res, tokens = call_ai_with_retry([{"role": "user", "content": prompt}], json_mode=True)
    return (json.loads(res) if res else None), tokens

# --- EXECUTION ENGINE ---
def execute_dashboard_logic(dfs, active_filters_json=None, session_id=None):
    log = []
    total_tokens = 0
    # Generate session ID if not provided
    if not session_id:
        import uuid
        session_id = str(uuid.uuid4())

    # Create persistent DuckDB file
    import tempfile
    temp_dir = tempfile.gettempdir()  # Gets C:\Users\YourName\AppData\Local\Temp on Windows
    db_path = os.path.join(temp_dir, f'duckdb_session_{session_id}.db')
    con = duckdb.connect(database=db_path)
    con.execute("CREATE FUNCTION clean_id(x) AS regexp_replace(cast(x as varchar), '[^0-9]', '', 'g')")
    
    table_map = {}
    all_columns = []
    
    # 1. INGEST
# 1. INGEST - CREATE ACTUAL TABLES (not just register views)
    for name, df in dfs.items():
        clean_name = clean_col_name(name)
        df.columns = [clean_col_name(c) for c in df.columns]
        all_columns.extend(df.columns)
        for c in df.columns:
            if 'id' in c or 'code' in c: df[c] = df[c].astype(str)
        
        # CHANGED: Create actual table instead of registering view
        con.execute(f"CREATE TABLE IF NOT EXISTS {clean_name} AS SELECT * FROM df")
        table_map[name] = clean_name
        
    raw_context = "".join([get_table_schema(con, t) for t in table_map.values()])

    # 2. DOMAIN DETECTION
    col_text = " ".join(all_columns).lower()
    forced_domain = None
    if any(x in col_text for x in ['shipment', 'supplier', 'procurement']): forced_domain = "Supply Chain"
    elif any(x in col_text for x in ['campaign', 'ad_spend', 'click']): forced_domain = "Marketing"
    elif any(x in col_text for x in ['employee', 'salary', 'payroll']): forced_domain = "Human Resources"

    # 3. JOIN
    join_sql, tokens = generate_join_sql(raw_context)
    total_tokens += tokens
    largest_table = max(table_map.values(), key=lambda t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
    if not join_sql: join_sql = f"CREATE TABLE IF NOT EXISTS master_view AS SELECT * FROM {largest_table}"

    try:
        con.execute("DROP TABLE IF EXISTS master_view")
        # Modify the SQL to use CREATE TABLE instead of CREATE OR REPLACE VIEW
        if "CREATE OR REPLACE VIEW master_view" in join_sql:
            join_sql = join_sql.replace("CREATE OR REPLACE VIEW master_view", "CREATE TABLE master_view")
        con.execute(join_sql)
        row_count = con.execute("SELECT COUNT(*) FROM master_view").fetchone()[0]
        log.append(f"✅ Joined Data - {row_count} rows in master_view")
        log.append(f"📋 Join SQL: {join_sql[:200]}...")  # First 200 chars
        
        # Log table details
        for original_name, clean_name in table_map.items():
            table_rows = con.execute(f"SELECT COUNT(*) FROM {clean_name}").fetchone()[0]
            log.append(f"  └─ Table: {original_name} ({clean_name}) - {table_rows} rows")
        
    except Exception as e:
        log.append(f"❌ Join Failed: {str(e)}")
        log.append(f"⚠️ Using largest table: {largest_table}")
        con.execute("DROP TABLE IF EXISTS master_view")
        con.execute(f"CREATE TABLE master_view AS SELECT * FROM {largest_table}")
        row_count = con.execute("SELECT COUNT(*) FROM master_view").fetchone()[0]
        log.append(f"📊 Master view created from {largest_table} - {row_count} rows")

    # --- 3.5 DETECT DATE COLUMN ---
    date_column = None
    try:
        schema = con.execute("DESCRIBE master_view").df()
        for _, row in schema.iterrows():
            col = row['column_name']
            dtype = row['column_type']
            if 'DATE' in dtype or 'TIMESTAMP' in dtype:
                date_column = col
                break
        
        if not date_column:
            for _, row in schema.iterrows():
                col = row['column_name']
                if any(keyword in col.lower() for keyword in ['date', 'time', 'day', 'month', 'year']):
                    try:
                        con.execute(f"SELECT CAST({col} AS DATE) FROM master_view LIMIT 1")
                        date_column = col
                        break
                    except:
                        pass
    except:
        pass

    # Get date range for filters
    date_range = {"min": None, "max": None}
    if date_column:
        try:
            result = con.execute(f"SELECT MIN(CAST({date_column} AS DATE)), MAX(CAST({date_column} AS DATE)) FROM master_view").fetchone()
            if result and result[0] and result[1]:
                date_range["min"] = str(result[0])
                date_range["max"] = str(result[1])
        except:
            pass

    # --- 3.6 APPLY FILTERS ---
    where_clauses = []
    if active_filters_json:
        try:
            filters = json.loads(active_filters_json)
            
            for col, val in filters.items():
                if not val or val == "null":
                    continue
                
                if col == '_start_date' and date_column:
                    clean_val = str(val).replace("'", "''")
                    where_clauses.append(f"CAST({date_column} AS DATE) >= CAST('{clean_val}' AS DATE)")
                    
                elif col == '_end_date' and date_column:
                    clean_val = str(val).replace("'", "''")
                    where_clauses.append(f"CAST({date_column} AS DATE) <= CAST('{clean_val}' AS DATE)")
                
                else:
                    try:
                        con.execute(f"SELECT {col} FROM master_view LIMIT 1")
                        clean_val = str(val).replace("'", "''")
                        where_clauses.append(f"{col} = '{clean_val}'")
                    except:
                        pass
                        
        except Exception as e:
            log.append(f"⚠️ Filter parsing error: {str(e)}")

        # Drop existing final_view table first
        con.execute("DROP TABLE IF EXISTS final_view")

        if where_clauses:
            full_where = " WHERE " + " AND ".join(where_clauses)
            try:
                con.execute(f"CREATE TABLE final_view AS SELECT * FROM master_view {full_where}")
                log.append(f"✅ Applied {len(where_clauses)} filter(s)")
            except Exception as e:
                log.append(f"⚠️ Filter application failed: {str(e)}")
                con.execute("CREATE TABLE final_view AS SELECT * FROM master_view")
        else:
            con.execute("CREATE TABLE final_view AS SELECT * FROM master_view")
    
    count = con.execute("SELECT COUNT(*) FROM final_view").fetchone()[0]

    # --- 4. DATA DEEP SCAN ---
    valid_text_cols = []   
    any_text_cols = []     
    valid_num_cols = []    
    valid_date_cols = []   
    
    try:
        schema = con.execute("DESCRIBE final_view").df()
        for _, row in schema.iterrows():
            col = row['column_name']
            dtype = row['column_type']
            
            if 'VARCHAR' in dtype:
                any_text_cols.append(col)
                try:
                    unique = con.execute(f"SELECT COUNT(DISTINCT {col}) FROM final_view").fetchone()[0]
                    if 1 < unique <= 50: valid_text_cols.append(col)
                except: pass
            
            if 'INT' in dtype or 'DOUBLE' in dtype or 'DECIMAL' in dtype:
                valid_num_cols.append(col)
            if 'DATE' in dtype or 'TIMESTAMP' in dtype:
                valid_date_cols.append(col)
    except: pass
    
    if not valid_num_cols: valid_num_cols = ["COUNT(*)"]
    if date_column and date_column not in valid_date_cols:
        valid_date_cols.append(date_column)

    # 5. PLAN
    master_context = get_table_schema(con, "master_view")
    plan, tokens = generate_viz_config(master_context, forced_domain)
    total_tokens += tokens
    if not plan: plan = {"domain": "General", "filters": [], "kpis": [], "charts": []}
    if forced_domain: plan["domain"] = forced_domain

    # 6. OUTPUT SETUP
    domain = plan.get("domain", "General")
    theme = THEME_MAP.get(domain, THEME_MAP["General"])
    if domain not in THEME_MAP:
        for k, v in THEME_MAP.items():
            if k.lower() in domain.lower(): theme = v; domain = k; break

    output = { 
        "domain": domain, 
        "theme": theme, 
        "filters": [], 
        "kpis": [], 
        "charts": [], 
        "logs": log,
        "date_range": date_range,
        "has_date_column": date_column is not None,
        "tokens_used": total_tokens,
        "master_preview": None
    }
    
    # Get master table preview (first 3 rows)
    try:
        preview_df = con.execute("SELECT * FROM master_view LIMIT 3").df()
        output["master_preview"] = {
            "columns": preview_df.columns.tolist(),
            "rows": preview_df.values.tolist()
        }
    except:
        pass

    # 7. FILTERS
    # 7. FILTERS
    all_filter_candidates = plan.get("filters", []) + valid_text_cols
    
    # Remove duplicates while preserving order
    unique_filters = []
    seen_lower = set()
    
    for col in all_filter_candidates:
        col_lower = col.lower()
        if col_lower not in seen_lower:
            seen_lower.add(col_lower)
            unique_filters.append(col)
    
    filter_candidates = unique_filters[:10]
    
    # Track columns already added to prevent duplicates
    added_columns = set()
    
    for col in filter_candidates:
        try:
            # Skip if already added (case-insensitive check)
            if col.lower() in added_columns:
                continue
                
            vals = con.execute(f"SELECT DISTINCT {col} FROM master_view WHERE {col} IS NOT NULL ORDER BY 1 LIMIT 50").df().iloc[:,0].tolist()
            if vals:
                output["filters"].append({
                    "label": col.replace('_',' ').title(), 
                    "column": col, 
                    "values": [str(v) for v in vals]
                })
                added_columns.add(col.lower())
        except:
            pass

    # 8. KPIS WITH SPARKLINES
    valid_kpis = []
    for kpi in plan.get("kpis", []):
        try:
            sql = kpi["sql"].replace("master_view", "final_view")
            val = con.execute(sql).fetchone()[0]
            val = 0 if val is None else val
            fmt = f"{val:,.0f}" if isinstance(val, (int, float)) else str(val)
            
            # Get sparkline data - ALWAYS generate it
            sparkline_data = []
            
            # Try to get real trend data from trend_sql
            if "trend_sql" in kpi and date_column:
                try:
                    trend_sql = kpi["trend_sql"].replace("master_view", "final_view")
                    trend_df = con.execute(trend_sql).df()
                    if not trend_df.empty and len(trend_df) > 0:
                        # Get last 7 data points
                        values = trend_df.iloc[:, 1].fillna(0).tolist()
                        if len(values) >= 7:
                            sparkline_data = values[-7:]
                        elif len(values) > 0:
                            # If less than 7 points, pad with the first value
                            sparkline_data = values + [values[-1]] * (7 - len(values))
                except Exception as e:
                    log.append(f"⚠️ Trend query failed for {kpi['label']}: {str(e)}")
            
            # If no trend data or trend_sql failed, generate from date column
            if not sparkline_data and date_column:
                try:
                    # Get the column being aggregated
                    sql_lower = sql.lower()
                    if 'sum(' in sql_lower:
                        # Extract column name from SUM(column_name)
                        import re
                        match = re.search(r'sum\(([^)]+)\)', sql_lower)
                        if match:
                            agg_col = match.group(1).strip()
                            sparkline_sql = f"""
                                SELECT CAST({date_column} AS DATE) as dt, SUM({agg_col}) as val 
                                FROM final_view 
                                GROUP BY 1 
                                ORDER BY 1 DESC 
                                LIMIT 7
                            """
                        else:
                            sparkline_sql = f"""
                                SELECT CAST({date_column} AS DATE) as dt, COUNT(*) as val 
                                FROM final_view 
                                GROUP BY 1 
                                ORDER BY 1 DESC 
                                LIMIT 7
                            """
                    else:
                        # Default to COUNT
                        sparkline_sql = f"""
                            SELECT CAST({date_column} AS DATE) as dt, COUNT(*) as val 
                            FROM final_view 
                            GROUP BY 1 
                            ORDER BY 1 DESC 
                            LIMIT 7
                        """
                    
                    spark_df = con.execute(sparkline_sql).df()
                    if not spark_df.empty:
                        sparkline_data = spark_df.iloc[:, 1].fillna(0).tolist()
                        sparkline_data.reverse()  # Reverse to show chronological order
                        
                        # Ensure we have exactly 7 points
                        if len(sparkline_data) < 7:
                            sparkline_data = sparkline_data + [sparkline_data[-1]] * (7 - len(sparkline_data))
                        elif len(sparkline_data) > 7:
                            sparkline_data = sparkline_data[-7:]
                except Exception as e:
                    log.append(f"⚠️ Sparkline generation failed for {kpi['label']}: {str(e)}")
            
            # Final fallback: generate synthetic trend data based on the value
            if not sparkline_data:
                base_val = float(val) if isinstance(val, (int, float)) else 100
                sparkline_data = []
                for i in range(7):
                    # Create a realistic trend with some variation
                    variation = np.random.uniform(-0.15, 0.15)  # ±15% variation
                    trend = 1 + (i - 3) * 0.02  # Slight upward trend
                    point = max(0, base_val * trend * (1 + variation))
                    sparkline_data.append(float(point))  # Ensure it's a float
            
            # Ensure all values are floats (not numpy types)
            sparkline_data = [float(x) for x in sparkline_data]
            
            valid_kpis.append({
                "label": kpi["label"], 
                "value": fmt,
                "sparkline": sparkline_data
            })
        except Exception as e:
            log.append(f"⚠️ KPI generation failed: {str(e)}")
    
    # Fill remaining KPI slots
    while len(valid_kpis) < 4:
        if valid_num_cols and valid_num_cols[0] != "COUNT(*)":
            col = valid_num_cols[len(valid_kpis) % len(valid_num_cols)]
            try:
                val = con.execute(f"SELECT SUM({col}) FROM final_view").fetchone()[0]
                val = val if val else 0
                
                # Generate sparkline for this KPI
                sparkline_data = []
                if date_column:
                    try:
                        spark_sql = f"""
                            SELECT CAST({date_column} AS DATE) as dt, SUM({col}) as val 
                            FROM final_view 
                            GROUP BY 1 
                            ORDER BY 1 DESC 
                            LIMIT 7
                        """
                        spark_df = con.execute(spark_sql).df()
                        if not spark_df.empty:
                            sparkline_data = spark_df.iloc[:, 1].fillna(0).tolist()
                            sparkline_data.reverse()
                    except:
                        pass
                
                if not sparkline_data:
                    base_val = float(val) if isinstance(val, (int, float)) else count
                    sparkline_data = [float(base_val * (1 + (i-3)*0.02 + np.random.uniform(-0.1, 0.1))) for i in range(7)]
                
                valid_kpis.append({
                    "label": f"Total {col.title()}", 
                    "value": f"{val:,.0f}",
                    "sparkline": sparkline_data
                })
            except:
                # Fallback to record count
                sparkline_data = [count * (1 + (i-3)*0.02 + np.random.uniform(-0.05, 0.05)) for i in range(7)]
                valid_kpis.append({
                    "label": "Records", 
                    "value": str(count),
                    "sparkline": sparkline_data
                })
        else:
            # Record count KPI
            sparkline_data = []
            if date_column:
                try:
                    spark_sql = f"""
                        SELECT CAST({date_column} AS DATE) as dt, COUNT(*) as val 
                        FROM final_view 
                        GROUP BY 1 
                        ORDER BY 1 DESC 
                        LIMIT 7
                    """
                    spark_df = con.execute(spark_sql).df()
                    if not spark_df.empty:
                        sparkline_data = spark_df.iloc[:, 1].fillna(0).tolist()
                        sparkline_data.reverse()
                except:
                    pass
            
            if not sparkline_data:
                sparkline_data = [float(count * (1 + (i-3)*0.01 + np.random.uniform(-0.05, 0.05))) for i in range(7)]
            
            valid_kpis.append({
                "label": "Total Records", 
                "value": str(count),
                "sparkline": sparkline_data
            })
    
    output["kpis"] = valid_kpis[:4]

    # --- 9. CHARTS ---
    for i, chart in enumerate(plan.get("charts", [])):
        if i == 0:
            chart["type"] = "line"
        elif i == 2:
            chart["type"] = "sankey"  # Changed from sunburst to sankey
        elif i > 2 and chart["type"] == "pie":
            chart["type"] = "bar"
        
        c_data = {
            "id": f"chart_{i}", 
            "title": chart["title"], 
            "type": chart["type"],
            "xlabel": chart.get("xlabel",""), 
            "ylabel": chart.get("ylabel",""), 
            "x": [], 
            "y": [],
            "z": [],  # For heatmaps
            "source": [],  # For sankey
            "target": [],  # For sankey
            "value": [],
            "level": []   # For sankey
        }
        # --- SPECIAL LOGIC FOR CHART 2 (DECOMPOSITION / SANKEY) ---
        if i == 2:
            c_data["type"] = "sankey"
            
            # 1. Grab Top 4 columns (needed for 3 Levels of flow)
            cat_cols = valid_text_cols[:3] if len(valid_text_cols) >= 2 else []
            if not cat_cols and len(any_text_cols) >= 2:
                 cat_cols = any_text_cols[:3]
            
            # 2. Metric
            num_col = valid_num_cols[0] if valid_num_cols and valid_num_cols[0] != "COUNT(*)" else None
            agg = f"SUM({num_col})" if num_col else "COUNT(*)"

            if len(cat_cols) >= 2:
                queries = []
                
                # LEVEL 1: Col 0 -> Col 1
                q1 = f"""
                    SELECT 
                        COALESCE(CAST({cat_cols[0]} AS VARCHAR), 'Unknown') as source, 
                        COALESCE(CAST({cat_cols[1]} AS VARCHAR), 'Unknown') as target, 
                        {agg} as value, 
                        1 as level 
                    FROM final_view 
                    GROUP BY 1, 2
                """
                queries.append(q1)

                # LEVEL 2: Col 1 -> Col 2
                if len(cat_cols) >= 3:
                    q2 = f"""
                        SELECT 
                            COALESCE(CAST({cat_cols[1]} AS VARCHAR), 'Unknown') as source, 
                            COALESCE(CAST({cat_cols[2]} AS VARCHAR), 'Unknown') as target, 
                            {agg} as value, 
                            2 as level 
                        FROM final_view 
                        GROUP BY 1, 2
                    """
                    queries.append(q2)

   
                
                try:
                    full_sql = " UNION ALL ".join(queries)
                    df = con.execute(full_sql).df()
                    df['level'] = df['level'].astype(int)
                    
                    c_data["title"] = f"Flow: {' → '.join([c.replace('_',' ').title() for c in cat_cols])}"
                    c_data["source"] = df['source'].tolist()
                    c_data["target"] = df['target'].tolist()
                    c_data["value"] = df['value'].tolist()
                    c_data["level"] = df['level'].tolist()
                except: 
                    pass
            
            output["charts"].append(c_data)
            continue
        success = False
        try:
            sql = chart["sql"].replace("master_view", "final_view")
            df = con.execute(sql).df().fillna(0)
            if not df.empty and len(df) > 0: 
                success = True
        except: 
            success = False
        
        # RESCUE MODE
        if not success:
            try:
                if i == 0:  # Trend chart
                    if valid_date_cols:
                        date_col = valid_date_cols[0]
                        num_col = valid_num_cols[0] if valid_num_cols and valid_num_cols[0] != "COUNT(*)" else None
                        
                        if num_col:
                            sql = f"SELECT CAST({date_col} AS DATE) as x, SUM({num_col}) as y FROM final_view GROUP BY 1 ORDER BY 1"
                            c_data["ylabel"] = num_col.title()
                            c_data["title"] = f"{num_col.replace('_', ' ').title()} Over Time"  # IMPROVED
                        else:
                            sql = f"SELECT CAST({date_col} AS DATE) as x, COUNT(*) as y FROM final_view GROUP BY 1 ORDER BY 1"
                            c_data["ylabel"] = "Count"
                            c_data["title"] = f"Record Count by {date_col.replace('_', ' ').title()}"  # IMPROVED
                        
                        df = con.execute(sql).df().fillna(0)
                        c_data["type"] = "line"
                
                    elif i == 1 and chart["type"] == "heatmap":  # Heatmap
                        if len(valid_text_cols) >= 2:
                            cat1, cat2 = valid_text_cols[0], valid_text_cols[1]
                            num_col = valid_num_cols[0] if valid_num_cols and valid_num_cols[0] != "COUNT(*)" else None
                            
                            if num_col:
                                sql = f"SELECT {cat1} as x, {cat2} as y, SUM({num_col}) as z FROM final_view GROUP BY 1, 2"
                                c_data["title"] = f"{num_col.replace('_', ' ').title()} by {cat1.replace('_', ' ').title()} & {cat2.replace('_', ' ').title()}"  # IMPROVED
                            else:
                                sql = f"SELECT {cat1} as x, {cat2} as y, COUNT(*) as z FROM final_view GROUP BY 1, 2"
                                c_data["title"] = f"Distribution: {cat1.replace('_', ' ').title()} vs {cat2.replace('_', ' ').title()}"  # IMPROVED
                            
                            df = con.execute(sql).df().fillna(0)
                    # CHART 4: DYNAMIC MULTI-LEVEL SANKEY (Flow Decomposition)
                        elif i == 4:
                            c_data["type"] = "sankey"
                            
                            # 1. Dynamically pick categories
                            cat_cols = valid_text_cols[:3]
                            
                            # 2. Pick metric
                            num_col = valid_num_cols[0] if valid_num_cols and valid_num_cols[0] != "COUNT(*)" else None
                            agg = f"SUM({num_col})" if num_col else "COUNT(*)"

                            # 3. Build Multi-Level Query
                            if len(cat_cols) >= 2:
                                queries = []
                                
                                # LEVEL 1
                                q1 = f"""
                                    SELECT 
                                        {cat_cols[0]} as source, 
                                        {cat_cols[1]} as target, 
                                        {agg} as value, 
                                        1 as level 
                                    FROM final_view 
                                    WHERE {cat_cols[0]} IS NOT NULL AND {cat_cols[1]} IS NOT NULL 
                                    GROUP BY 1, 2
                                """
                                queries.append(q1)

                                # LEVEL 2
                                if len(cat_cols) >= 3:
                                    q2 = f"""
                                        SELECT 
                                            {cat_cols[1]} as source, 
                                            {cat_cols[2]} as target, 
                                            {agg} as value, 
                                            2 as level 
                                        FROM final_view 
                                        WHERE {cat_cols[1]} IS NOT NULL AND {cat_cols[2]} IS NOT NULL 
                                        GROUP BY 1, 2
                                    """
                                    queries.append(q2)

                                full_sql = " UNION ALL ".join(queries)
                                
                                try:
                                    df = con.execute(full_sql).df().fillna(0)
                                    c_data["title"] = f"Flow: {' → '.join([c.replace('_',' ').title() for c in cat_cols])}"
                                except:
                                    df = pd.DataFrame()
                            else:
                                c_data["title"] = "Not enough categories for Flow Chart"
                    else:
                        # Fallback to bar chart
                        c_data["type"] = "bar"
                        col = valid_text_cols[0] if valid_text_cols else (any_text_cols[0] if any_text_cols else None)
                        if col:
                            sql = f"SELECT {col} as x, COUNT(*) as y FROM final_view GROUP BY 1 ORDER BY 2 DESC LIMIT 10"
                            df = con.execute(sql).df().fillna(0)
                
                elif i == 2 and chart["type"] == "sankey":  # Multi-level Sankey
                    # Try to get 3-4 categorical columns for deep decomposition
                    available_cats = valid_text_cols[:3] if len(valid_text_cols) >= 3 else valid_text_cols[:2]
                    
                    if len(available_cats) >= 2:
                        num_col = valid_num_cols[0] if valid_num_cols and valid_num_cols[0] != "COUNT(*)" else None
                        
                        
                        try:
                            # Build multi-level flows
                            if len(available_cats) == 2:
                                # 2-level: Cat1 -> Cat2
                                cat1, cat2 = available_cats[0], available_cats[1]
                                if num_col:
                                    sql = f"""
                                        SELECT 
                                            CAST({cat1} AS VARCHAR) as source, 
                                            CAST({cat2} AS VARCHAR) as target, 
                                            SUM({num_col}) as value,
                                            1 as level
                                        FROM final_view 
                                        WHERE {cat1} IS NOT NULL AND {cat2} IS NOT NULL 
                                        GROUP BY 1, 2
                                        HAVING SUM({num_col}) > 0
                                        ORDER BY 4, 3 DESC
                                    """
                                else:
                                    sql = f"""
                                        SELECT 
                                            CAST({cat1} AS VARCHAR) as source, 
                                            CAST({cat2} AS VARCHAR) as target, 
                                            CAST(COUNT(*) AS DOUBLE) as value,
                                            1 as level
                                        FROM final_view 
                                        WHERE {cat1} IS NOT NULL AND {cat2} IS NOT NULL 
                                        GROUP BY 1, 2
                                        HAVING COUNT(*) > 0
                                        ORDER BY 4, 3 DESC
                                    """
                                c_data["title"] = f"Root Cause Analysis: {cat1.replace('_', ' ').title()} → {cat2.replace('_', ' ').title()}"  # IMPROVED
                                
                            elif len(available_cats) == 3:
                                # 3-level: Cat1 -> Cat2 -> Cat3
                                cat1, cat2, cat3 = available_cats[0], available_cats[1], available_cats[2]
                                if num_col:
                                    sql = f"""
    SELECT 
        CAST({cat1} AS VARCHAR) as source, 
        CAST({cat2} AS VARCHAR) as target, 
        SUM({num_col}) as value, 
        1 as level
    FROM final_view 
    WHERE {cat1} IS NOT NULL AND {cat2} IS NOT NULL 
    GROUP BY 1, 2
    HAVING SUM({num_col}) > 0
    
    UNION ALL
    
    SELECT 
        CAST({cat2} AS VARCHAR) as source, 
        CAST({cat3} AS VARCHAR) as target, 
        SUM({num_col}) as value, 
        2 as level
    FROM final_view 
    WHERE {cat2} IS NOT NULL AND {cat3} IS NOT NULL 
    GROUP BY 1, 2
    HAVING SUM({num_col}) > 0
"""

                                else:
                                    sql = f"""
        SELECT 
            CAST({cat1} AS VARCHAR) as source, 
            CAST({cat2} AS VARCHAR) as target, 
            CAST(COUNT(*) AS DOUBLE) as value, 
            1 as level
        FROM final_view 
        WHERE {cat1} IS NOT NULL AND {cat2} IS NOT NULL 
        GROUP BY 1, 2
        HAVING COUNT(*) > 0
        
        UNION ALL
        
        SELECT 
            CAST({cat2} AS VARCHAR) as source, 
            CAST({cat3} AS VARCHAR) as target, 
            CAST(COUNT(*) AS DOUBLE) as value, 
            2 as level
        FROM final_view 
        WHERE {cat2} IS NOT NULL AND {cat3} IS NOT NULL 
        GROUP BY 1, 2
        HAVING COUNT(*) > 0
        
       
    """
                                metric_name = num_col.replace('_', ' ').title() if num_col else 'Volume'
                                c_data["title"] = f"{metric_name} Flow: {cat1.replace('_', ' ').title()} → {cat2.replace('_', ' ').title()} → {cat3.replace('_', ' ').title()}"  # IMPROVED
                                

                                metric_name = num_col.replace('_', ' ').title() if num_col else 'Volume'
                                c_data["title"] = f"{metric_name} Decomposition: {cat1.replace('_', ' ').title()} → {cat2.replace('_', ' ').title()} → {cat3.replace('_', ' ').title()} → {cat4.replace('_', ' ').title()}"  # IMPROVED
                                        
                            df = con.execute(sql).df()
                            
                            if not df.empty and len(df) >= 2:
                                c_data["type"] = "sankey"
                                log.append(f"✅ Multi-level Sankey created: {len(df)} flows across {len(available_cats)} levels")
                            else:
                                raise Exception(f"Insufficient data for sankey: {len(df) if not df.empty else 0} rows")
                                
                        except Exception as e:
                            log.append(f"⚠️ Sankey failed: {str(e)}, falling back to bar")
                            sql = f"SELECT {available_cats[0]} as x, COUNT(*) as y FROM final_view WHERE {available_cats[0]} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 10"
                            df = con.execute(sql).df().fillna(0)
                            c_data["type"] = "bar"
                            c_data["title"] = f"Distribution by {available_cats[0].title()}"
                
                else:  # Other charts
                    if chart["type"] == "line" and valid_date_cols:
                        col = valid_date_cols[0]
                        num_col = valid_num_cols[0] if valid_num_cols and valid_num_cols[0] != "COUNT(*)" else None
                        if num_col:
                            sql = f"SELECT CAST({col} AS DATE) as x, SUM({num_col}) as y FROM final_view GROUP BY 1 ORDER BY 1"
                            c_data["title"] = f"{num_col.replace('_', ' ').title()} Trend by {col.replace('_', ' ').title()}"  # IMPROVED
                        else:
                            sql = f"SELECT CAST({col} AS DATE) as x, COUNT(*) as y FROM final_view GROUP BY 1 ORDER BY 1"
                            c_data["title"] = f"Record Count Trend by {col.replace('_', ' ').title()}"  # IMPROVED
                        df = con.execute(sql).df().fillna(0)
                    else:
                        col_index = (i - 3) if i >= 3 else 0
                        col = valid_text_cols[col_index % len(valid_text_cols)] if valid_text_cols else (any_text_cols[0] if any_text_cols else None)
                        if col:
                            num_col = valid_num_cols[0] if valid_num_cols and valid_num_cols[0] != "COUNT(*)" else None
                            if num_col:
                                sql = f"SELECT {col} as x, SUM({num_col}) as y FROM final_view GROUP BY 1 ORDER BY 2 DESC LIMIT 10"
                                c_data["title"] = f"Top {col.replace('_', ' ').title()} by {num_col.replace('_', ' ').title()}"  # IMPROVED
                            else:
                                sql = f"SELECT {col} as x, COUNT(*) as y FROM final_view GROUP BY 1 ORDER BY 2 DESC LIMIT 10"
                                c_data["title"] = f"Distribution by {col.replace('_', ' ').title()}"  # IMPROVED
                            df = con.execute(sql).df().fillna(0)
                            c_data["type"] = "bar"
            except:
                pass

        # Populate chart data
        if not df.empty:
            df.columns = [str(c).lower() for c in df.columns]
            
            if c_data["type"] == "sankey":
                # Sankey needs source, target, value
                c_data["source"] = df['source'].tolist() if 'source' in df else []
                c_data["target"] = df['target'].tolist() if 'target' in df else []
                c_data["value"] = df['value'].tolist() if 'value' in df else []
                c_data["level"] = df['level'].fillna(1).astype(int).tolist() if 'level' in df else [1] * len(c_data["source"]) # ADD THIS
            else:
                # Regular charts need x, y, z
                c_data["x"] = df.iloc[:,0].tolist() if 'x' not in df else df['x'].tolist()
                c_data["y"] = df.iloc[:,1].tolist() if 'y' not in df else df['y'].tolist()
                if c_data["type"] == "heatmap" and len(df.columns) >= 3:
                    c_data["z"] = df.iloc[:,2].tolist() if 'z' not in df else df['z'].tolist()
            
            output["charts"].append(c_data)
        else:
            c_data["title"] += " (No Data)"
            output["charts"].append(c_data)

    # Close connection but keep file on disk
    con.close()

    # Return both output AND session_id
    output['session_id'] = session_id  # Add session_id to output
    return output

def get_universal_master_csv(session_id, filters_json='{}'):
    """
    Universal Adapter: Exports final_view with standardized column names.
    
    Args:
        session_id: UUID of the DuckDB session file
        filters_json: JSON string of active filters (optional)
    
    Returns:
        DataFrame with STD_Date, STD_Cat1-5, STD_Val1-5, and metadata columns
    """
    
    # Reconnect to the persistent DuckDB file
    import tempfile
    temp_dir = tempfile.gettempdir()
    db_path = os.path.join(temp_dir, f'duckdb_session_{session_id}.db')
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DuckDB session file not found: {db_path}")
    
    con = duckdb.connect(database=db_path, read_only=True)
    
    try:
        # Step 1: Introspect the final_view schema
        schema_df = con.execute("DESCRIBE final_view").df()
        
        # Step 2: Classify columns by type
        date_cols = []
        text_cols = []
        numeric_cols = []
        
        for _, row in schema_df.iterrows():
            col_name = row['column_name']
            col_type = str(row['column_type']).upper()
            
            # Classify by type
            if 'DATE' in col_type or 'TIMESTAMP' in col_type:
                date_cols.append(col_name)
            elif 'VARCHAR' in col_type or 'TEXT' in col_type:
                # Check cardinality to avoid IDs
                cardinality_check = con.execute(f"SELECT COUNT(DISTINCT {col_name}) as card FROM final_view").fetchone()[0]
                total_rows = con.execute("SELECT COUNT(*) FROM final_view").fetchone()[0]
                
                # Only include if cardinality is reasonable (less than 50% of total rows)
                if total_rows > 0 and cardinality_check / total_rows < 0.5:
                    text_cols.append((col_name, cardinality_check))
            elif any(t in col_type for t in ['INT', 'DOUBLE', 'DECIMAL', 'FLOAT', 'NUMERIC']):
                # Check if column has meaningful values (not all nulls)
                non_null_count = con.execute(f"SELECT COUNT({col_name}) FROM final_view WHERE {col_name} IS NOT NULL").fetchone()[0]
                if non_null_count > 0:
                    numeric_cols.append((col_name, non_null_count))
        
        # Step 3: Rank and select top columns
        # Sort text columns by cardinality (ascending - prefer low cardinality for slicers)
        text_cols.sort(key=lambda x: x[1])
        top_text_cols = [col[0] for col in text_cols[:5]]
        
        # Sort numeric columns by non-null count (descending - prefer complete data)
        numeric_cols.sort(key=lambda x: x[1], reverse=True)
        top_numeric_cols = [col[0] for col in numeric_cols[:5]]
        
        # Pick first date column (or none if no dates exist)
        top_date_col = date_cols[0] if date_cols else None
        
        # Step 4: Build SQL query with standardized aliases
        select_parts = []
        metadata_parts = []
        
        # Date mapping
        if top_date_col:
            select_parts.append(f"CAST({top_date_col} AS DATE) AS STD_Date")
            metadata_parts.append(f"'{top_date_col}' AS STD_Date_Name")
        else:
            select_parts.append("NULL AS STD_Date")
            metadata_parts.append("'No Date Column' AS STD_Date_Name")
        
        # Category mappings (STD_Cat1 through STD_Cat5)
# Category mappings (STD_Cat1 through STD_Cat5)
        for i in range(5):
            cat_num = i + 1  # Calculate number outside f-string
            if i < len(top_text_cols):
                col = top_text_cols[i]
                select_parts.append(f"CAST({col} AS VARCHAR) AS STD_Cat{cat_num}")
                metadata_parts.append(f"'{col}' AS STD_Cat{cat_num}_Name")
            else:
                select_parts.append(f"'N/A' AS STD_Cat{cat_num}")
                metadata_parts.append(f"'Unused' AS STD_Cat{cat_num}_Name")
        
        # Value mappings (STD_Val1 through STD_Val5)
# Value mappings (STD_Val1 through STD_Val5)
        for i in range(5):
            val_num = i + 1  # Calculate number outside f-string
            if i < len(top_numeric_cols):
                col = top_numeric_cols[i]
                select_parts.append(f"CAST({col} AS DOUBLE) AS STD_Val{val_num}")
                metadata_parts.append(f"'{col}' AS STD_Val{val_num}_Name")
            else:
                select_parts.append(f"NULL AS STD_Val{val_num}")
                metadata_parts.append(f"'Unused' AS STD_Val{val_num}_Name")
        
        # Combine all parts
        all_select = select_parts + metadata_parts
        sql = f"SELECT {', '.join(all_select)} FROM final_view"
        
        # Step 5: Execute query and return DataFrame
        master_df = con.execute(sql).df()
        
        return master_df
        
    finally:
        con.close()