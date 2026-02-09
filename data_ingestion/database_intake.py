import psycopg2

def connect_to_db():
    # Connect to your postgres DB
    conn = psycopg2.connect(
        host="localhost",
        database="postgres",
        user="postgres",
        password="postgres"
    )
    return conn

def main():
    conn = connect_to_db()
    print("Connected to the database successfully!")
    conn.close()

if __name__ == "__main__":
    main()
