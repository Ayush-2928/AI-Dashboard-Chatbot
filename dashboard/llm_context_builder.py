def get_table_schema(con, table_name, include_sample=True):
    try:
        info = con.execute(f"DESCRIBE {table_name}").df()
        cols = ", ".join([f"{r['column_name']} ({r['column_type']})" for _, r in info.iterrows()])
        if include_sample:
            sample = con.execute(f"SELECT * FROM {table_name} LIMIT 2").df().to_string(index=False)
            return f"TABLE: {table_name}\nCOLUMNS: {cols}\nSAMPLE:\n{sample}\n" + "-" * 30 + "\n"
        return f"TABLE: {table_name}\nCOLUMNS: {cols}\nSAMPLE: [omitted - metadata only mode]\n" + "-" * 30 + "\n"
    except Exception:
        return ""


def build_schema_context_from_columns(table_name, columns, include_sample=True, sample_df=None):
    cols_text = ", ".join([f"{name} ({dtype})" for name, dtype in columns])
    if include_sample and sample_df is not None and not sample_df.empty:
        sample_text = sample_df.head(2).to_string(index=False)
        return f"TABLE: {table_name}\nCOLUMNS: {cols_text}\nSAMPLE:\n{sample_text}\n" + "-" * 30 + "\n"
    return f"TABLE: {table_name}\nCOLUMNS: {cols_text}\nSAMPLE: [omitted - metadata only mode]\n" + "-" * 30 + "\n"
