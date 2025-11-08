import os
import psycopg2
from psycopg2 import pool
import pandas as pd
from config import Config

class DatabaseConnection:
    def __init__(self):
        self.pool = None
        self._initialized = False

    def init_pool(self):
        """Initialize connection pool (lazy initialization)"""
        if self._initialized and self.pool:
            return
        
        try:
            # Support connection string (for Supabase pooler)
            connection_string = os.getenv('DATABASE_URL')
            if connection_string:
                # Parse connection string
                import urllib.parse
                parsed = urllib.parse.urlparse(connection_string)
                
                # Parse query parameters
                query_params = urllib.parse.parse_qs(parsed.query)
                sslmode = query_params.get('sslmode', ['prefer'])[0]
                options = query_params.get('options', [None])[0]
                
                # Build connection parameters
                conn_params = {
                    'host': parsed.hostname,
                    'port': parsed.port or 5432,
                    'database': parsed.path[1:] if parsed.path.startswith('/') else parsed.path,  # Remove leading '/'
                    'user': parsed.username,
                    'password': urllib.parse.unquote(parsed.password),  # Decode URL-encoded password
                    'sslmode': sslmode
                }
                
                # Add options if present (for Supabase project identifier)
                if options:
                    conn_params['options'] = options
                
                # For Supabase, require SSL if not specified
                if 'supabase' in parsed.hostname and 'sslmode' not in connection_string:
                    conn_params['sslmode'] = 'require'
                
                print(f"Connecting to database: {parsed.hostname}:{conn_params['port']}")
                self.pool = psycopg2.pool.SimpleConnectionPool(1, 10, **conn_params)
            else:
                # Use individual config
                conn_params = {
                    'host': Config.DB_HOST,
                    'port': Config.DB_PORT,
                    'database': Config.DB_NAME,
                    'user': Config.DB_USER,
                    'password': Config.DB_PASSWORD
                }
                
                # For Supabase, require SSL
                if 'supabase.co' in Config.DB_HOST:
                    conn_params['sslmode'] = 'require'
                
                self.pool = psycopg2.pool.SimpleConnectionPool(1, 10, **conn_params)
            if self.pool:
                print("Connection pool created successfully")
                self._initialized = True
        except Exception as e:
            print(f"Error creating connection pool: {e}")
            print(f"Host: {Config.DB_HOST}, Port: {Config.DB_PORT}")
            raise
    
    def _ensure_pool(self):
        """Ensure connection pool is initialized"""
        if not self._initialized or not self.pool:
            self.init_pool()

    def get_connection(self):
        self._ensure_pool()
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
        self._ensure_pool()
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
        self._ensure_pool()
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
