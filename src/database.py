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
                    'sslmode': sslmode
                }
                
                # Handle password (may be None or URL-encoded)
                if parsed.password:
                    conn_params['password'] = urllib.parse.unquote(parsed.password)  # Decode URL-encoded password
                elif parsed.username:
                    # Password might be in username:password format
                    if ':' in parsed.username:
                        user, password = parsed.username.rsplit(':', 1)
                        conn_params['user'] = user
                        conn_params['password'] = urllib.parse.unquote(password)
                
                # Add options if present (for Supabase project identifier)
                if options:
                    conn_params['options'] = options
                
                # For Supabase, require SSL if not specified
                if parsed.hostname and 'supabase' in parsed.hostname and 'sslmode' not in connection_string:
                    conn_params['sslmode'] = 'require'
                
                # Validate required parameters
                if not conn_params.get('host') or not conn_params.get('database'):
                    raise ValueError(f"Invalid DATABASE_URL: missing host or database. Host: {conn_params.get('host')}, Database: {conn_params.get('database')}")
                
                print(f"Connecting to database: {conn_params.get('host')}:{conn_params.get('port')}")
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
            connection_string = os.getenv('DATABASE_URL')
            if connection_string:
                print(f"DATABASE_URL is set (length: {len(connection_string)})")
                # Show first and last 30 chars for debugging (hide password)
                if len(connection_string) > 60:
                    print(f"DATABASE_URL preview: {connection_string[:30]}...{connection_string[-30:]}")
                else:
                    print(f"DATABASE_URL: {connection_string}")
            else:
                print(f"DATABASE_URL is NOT set. Using individual config:")
                print(f"Host: {Config.DB_HOST}, Port: {Config.DB_PORT}, DB: {Config.DB_NAME}, User: {Config.DB_USER}")
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
        """Return connection to pool, handling edge cases"""
        if conn is None:
            return
        
        if self.pool is None:
            # Pool not initialized, just close the connection
            try:
                conn.close()
            except:
                pass
            return
        
        try:
            # Check if connection is still valid
            if conn.closed != 0:
                # Connection already closed, don't try to return it
                return
            
            # Return connection to pool
            self.pool.putconn(conn)
        except psycopg2.pool.PoolError as e:
            # Connection not from this pool or already returned
            # Just close it instead of raising error
            try:
                conn.close()
            except:
                pass
            # Don't print error for unkeyed connections (expected in some edge cases)
            pass
        except Exception as e:
            # Other errors - try to close connection
            try:
                conn.close()
            except:
                pass
            # Only print if it's not a pool-related error
            if "unkeyed" not in str(e).lower():
                print(f"Warning: Error returning connection to pool: {e}")

    def execute_query(self, query, params=None):
        self._ensure_pool()
        conn = self.get_connection()
        try:
            # Suppress pandas warning about DBAPI2 connections
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, message='.*pandas only supports SQLAlchemy.*')
            df = pd.read_sql_query(query, conn, params=params)
            return df
        except Exception as e:
            print(f"Error executing query: {e}")
            raise
        finally:
            # Always try to return connection to pool
            if conn is not None:
                try:
                    # Close any open cursors first
                    if conn.closed == 0:
                        # Reset connection state before returning to pool
                        conn.rollback()
                except:
                    pass  # Connection might already be closed
                
                # Return connection to pool (handles errors internally)
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
            # Return connection to pool (handles errors internally)
            if conn is not None:
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
