import os
import psycopg
from dotenv import load_dotenv

def main():
    # Load environment variables from .env
    load_dotenv()

    # Get credentials for the knowledge base database
    # Stripping space/quotes just in case the .env format is irregular
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
        print(f"DEBUG: dbname={dbname}, user={user}, password={'***' if password else None}, host={host}")
        return

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
                print("\nFetching all 'name' entries from 'pgadmin_module' table...\n")
                
                # Selecting id and name to make it useful for mapping
                cur.execute('SELECT id, name FROM pgadmin_module ORDER BY id;')
                rows = cur.fetchall()

                if not rows:
                    print("No entries found in 'pgadmin_module'.")
                else:
                    print(f"{'ID':<5} | {'MODULE NAME'}")
                    print("-" * 30)
                    for row in rows:
                        print(f"{row[0]:<5} | {row[1]}")
                
    except Exception as e:
        print(f"Database error: {e}")

if __name__ == "__main__":
    main()
