import os
import psycopg
import json
from dotenv import load_dotenv

def main():
    # Load environment variables from .env
    load_dotenv()

    # Get credentials for the knowledge base database
    def get_env_var(var_name, default=None):
        val = os.getenv(var_name, default)
        if val:
            return val.strip().strip("'").strip('"')
        return default

    dbname = get_env_var("DATA_DB_NAME")
    user = get_env_var("DATA_DB_USER")
    password = get_env_var("DATA_DB_PASSWORD")
    host = get_env_var("DATA_DB_HOST")
    port = get_env_var("DATA_DB_PORT", "5432")
    sslmode = get_env_var("DATA_DB_SSLMODE", "require")

    if not all([dbname, user, password, host]):
        print("Error: Missing database credentials in .env file.")
        return

    module_name = "GT Sales Module"

    try:
        print(f"Connecting to database: {host}:{port}/{dbname}...")
        with psycopg.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port,
            sslmode=sslmode
        ) as conn:
            with conn.cursor() as cur:
                print(f"\nFetching ALL columns for module: '{module_name}'...\n")
                
                # Fetch all columns for the specific module
                cur.execute('SELECT * FROM pgadmin_module WHERE name = %s;', (module_name,))
                row = cur.fetchone()

                if not row:
                    print(f"No module found with name '{module_name}'.")
                    # Let's list what *is* there just in case of typos
                    cur.execute('SELECT name FROM pgadmin_module LIMIT 10;')
                    others = [r[0] for r in cur.fetchall()]
                    print(f"Other available modules: {', '.join(others)}")
                else:
                    # Get column names
                    col_names = [desc[0] for desc in cur.description]
                    
                    print(f"{'='*80}")
                    print(f" MODULE DETAILS: {module_name} ")
                    print(f"{'='*80}")

                    for col, val in zip(col_names, row):
                        print(f"\n--- {col.upper()} ---")
                        if isinstance(val, (dict, list)):
                            print(json.dumps(val, indent=2))
                        elif isinstance(val, str) and (val.startswith('{') or val.startswith('[')):
                            try:
                                # Attempt to parse string as JSON if it looks like one
                                parsed = json.loads(val)
                                print(json.dumps(parsed, indent=2))
                            except:
                                print(val)
                        else:
                            print(val)
                
    except Exception as e:
        print(f"Database error: {e}")

if __name__ == "__main__":
    main()
