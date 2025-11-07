import psycopg2
from psycopg2 import pool
import pandas as pd
from config import Config

class DatabaseConnection:
    def __init__(self):
        self.pool = None
        self.init_pool()

    def init_pool(self):
        try:
            self.pool = psycopg2.pool.SimpleConnectionPool(
                1, 10,  # minconn, maxconn, update as needed
                host=Config.DB_HOST,
                port=Config.DB_PORT,
                database=Config.DB_NAME,
                user=Config.DB_USER,
                password=Config.DB_PASSWORD
            )
            if self.pool:
                print("Connection pool created successfully")
        except Exception as e:
            print(f"Error creating connection pool: {e}")
            raise

    def get_connection(self):
        try:
            return self.pool.getconn()
        except Exception as e:
            print(f"Error getting connection from pool: {e}")
            raise

    def put_connection(self, conn):
        try:
            self.pool.putconn(conn)
        except Exception as e:
            print(f"Error returning connection to pool: {e}")
            raise

    def execute_query(self, query, params=None):
        conn = self.get_connection()
        try:
            df = pd.read_sql_query(query, conn, params=params)
            return df
        except Exception as e:
            print(f"Error executing query: {e}")
            raise
        finally:
            self.put_connection(conn)

    def execute_insert(self, query, params=None):
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            cursor.close()
        except Exception as e:
            print(f"Error executing insert: {e}")
            conn.rollback()
            raise
        finally:
            self.put_connection(conn)

    def close(self):
        if self.pool:
            self.pool.closeall()

# Global database instance
if 'db' in globals():
    try:
        db.close()
    except Exception:
        pass

db = DatabaseConnection()
