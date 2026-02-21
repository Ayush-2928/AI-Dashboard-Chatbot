import os
import re

DEFAULT_FORBIDDEN_SQL_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
    "TRUNCATE", "ATTACH", "DETACH", "COPY", "PRAGMA", "CALL",
    "EXPORT", "IMPORT"
}


def redact_sensitive_text(text):
    value = str(text or "")
    secret_values = [
        os.getenv("OPENAI_API_KEY", ""),
        os.getenv("DATABRICKS_ACCESS_TOKEN", ""),
        os.getenv("DATABRICKS_TOKEN", "")
    ]
    for secret in secret_values:
        if secret:
            value = value.replace(secret, "[REDACTED]")

    value = re.sub(r"dapi[0-9A-Za-z\-_]{8,}", "dapi[REDACTED]", value)
    value = re.sub(r"sk-[0-9A-Za-z\-_]{12,}", "sk-[REDACTED]", value)
    return value


def apply_sql_security_and_cost_guardrails(
    sql,
    forbidden_sql_keywords=None,
    databricks_mode=False,
    strict_guardrails=True,
    max_limit=5000,
):
    if not sql:
        return sql, []

    cleaned = sql.replace("```sql", "").replace("```", "").strip().rstrip(";")
    upper_sql = cleaned.upper()
    notes = []

    if not (upper_sql.startswith("SELECT") or upper_sql.startswith("WITH")):
        raise ValueError("Only SELECT/WITH queries are allowed")

    keyword_set = forbidden_sql_keywords or DEFAULT_FORBIDDEN_SQL_KEYWORDS
    for keyword in keyword_set:
        if re.search(rf"\b{keyword}\b", upper_sql):
            raise ValueError(f"Forbidden SQL keyword detected: {keyword}")

    if databricks_mode and strict_guardrails:
        has_limit = re.search(r"\bLIMIT\s+\d+\b", upper_sql) is not None
        has_group = re.search(r"\bGROUP\s+BY\b", upper_sql) is not None
        has_agg = re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", upper_sql) is not None
        select_star = re.search(r"\bSELECT\s+\*\b", upper_sql) is not None

        if select_star and not has_limit:
            cleaned += f" LIMIT {max_limit}"
            notes.append(f"Applied cost guardrail: SELECT * capped with LIMIT {max_limit}")
            upper_sql = cleaned.upper()
            has_limit = True

        if (not has_limit) and (not has_group) and (not has_agg):
            cleaned += f" LIMIT {max_limit}"
            notes.append(f"Applied cost guardrail: non-aggregated query capped with LIMIT {max_limit}")
            upper_sql = cleaned.upper()

        limit_match = re.search(r"\bLIMIT\s+(\d+)\b", upper_sql)
        if limit_match:
            requested_limit = int(limit_match.group(1))
            if requested_limit > max_limit:
                cleaned = re.sub(
                    r"\bLIMIT\s+\d+\b",
                    f"LIMIT {max_limit}",
                    cleaned,
                    count=1,
                    flags=re.IGNORECASE,
                )
                notes.append(f"Applied cost guardrail: LIMIT reduced from {requested_limit} to {max_limit}")

    return cleaned, notes
