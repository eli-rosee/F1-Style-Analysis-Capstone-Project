import psycopg2
from psycopg2 import Error

def connect_to_db():
    # Connect to your postgres DB
    conn = psycopg2.connect(
        host="localhost",
        database="postgres",
        user="postgres",
        password="Team6"
    )
    return conn

def main():
    try:
        conn = connect_to_db()
        print("Connected to the database successfully!")
        cursor = conn.cursor()
        print("PostgreSQL server information")
        print(conn.get_dsn_parameters(), "\n")
        # Executing a SQL query
        cursor.execute("SELECT version();")
        # Fetch result
        record = cursor.fetchone()
        print("You are connected to - ", record, "\n")

    except (Exception, Error) as e:
        print(f"Error connecting to the database: {e}")
    finally:
        if(conn):
            conn.close()
            cursor.close()
            print("PostgreSQL connection is closed")

if __name__ == "__main__":
    main()
