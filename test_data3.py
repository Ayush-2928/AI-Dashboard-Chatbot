import os
import psycopg
from dotenv import load_dotenv

def main():
    # Load environment variables from .env
    load_dotenv()

    # Database connection parameters for the secondary DB
    dbname = os.getenv("DATA_DB_NAME")
    user = os.getenv("DATA_DB_USER")
    password = os.getenv("DATA_DB_PASSWORD")
    host = os.getenv("DATA_DB_HOST")
    port = os.getenv("DATA_DB_PORT", "5432")
    sslmode = os.getenv("DATA_DB_SSLMODE", "require")

    if not all([dbname, user, password, host]):
        print("Error: Missing database credentials in .env file.")
        print("Please ensure DATA_DB_NAME, DATA_DB_USER, DATA_DB_PASSWORD, and DATA_DB_HOST are set.")
        return

    # Construct connection string
    # Psycopg 3 connection string or keyword arguments
    try:
        print(f"Connecting to secondary database: {host}:{port}/{dbname}...")
        
        # Connect to the database
        with psycopg.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port,
            sslmode=sslmode
        ) as conn:
            with conn.cursor() as cur:
                # 1. Get all tables in the public schema
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_type = 'BASE TABLE'
                    ORDER BY table_name;
                """)
                tables = [row[0] for row in cur.fetchall()]

                if not tables:
                    print("No tables found in the 'public' schema.")
                    return

                print(f"Found {len(tables)} tables in 'public' schema.\n")

                for table in tables:
                    print(f"{'='*80}")
                    print(f" TABLE: {table.upper()} ")
                    print(f"{'='*80}")

                    # 2. Get columns for the table
                    cur.execute("""
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name = %s 
                        AND table_schema = 'public'
                        ORDER BY ordinal_position;
                    """, (table,))
                    columns = cur.fetchall()
                    
                    column_names = [col[0] for col in columns]
                    print(f"COLUMNS ({len(column_names)}):")
                    print(f"  {', '.join(column_names)}")
                    print()

                    # 3. Get top 10 rows
                    print("TOP 2 ROWS:")
                    try:
                        # Using double quotes for table name in case of reserved words or special characters
                        cur.execute(f'SELECT * FROM "{table}" LIMIT 2;')
                        rows = cur.fetchall()
                        
                        if not rows:
                            print("  (Table is empty)")
                        else:
                            # Calculate column widths for basic formatting
                            # We'll just print a few rows in a readable way
                            for i, row in enumerate(rows, 1):
                                print(f"  Row {i}:")
                                for col_name, value in zip(column_names, row):
                                    print(f"    {col_name:20}: {value}")
                                print("-" * 40)
                                
                    except Exception as e:
                        print(f"  Error fetching rows from {table}: {e}")
                    
                    print("\n" * 2)

    except Exception as e:
        print(f"Failed to connect or query the database: {e}")

if __name__ == "__main__":
    main()
