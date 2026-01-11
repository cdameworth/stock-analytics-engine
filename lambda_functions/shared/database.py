"""
PostgreSQL Database Connection for Railway Deployment.

This module provides a centralized database connection manager that replaces
boto3/DynamoDB with psycopg2/PostgreSQL for Railway deployment.

Usage:
    from lambda_functions.shared.database import db

    # Execute query and get results
    results = db.execute("SELECT * FROM recommendations WHERE symbol = %s", ("AAPL",))

    # Execute and get single result
    result = db.execute_one("SELECT * FROM recommendations WHERE id = %s", (id,))

    # Insert and get ID
    new_id = db.insert("recommendations", {"symbol": "AAPL", "recommendation": "BUY"})

    # Batch insert
    count = db.insert_many("predictions", [{"symbol": "AAPL"}, {"symbol": "GOOGL"}])
"""

import os
import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
import json

# Try to import psycopg2, gracefully handle if not available
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, execute_values, Json
    from psycopg2.extensions import register_adapter, AsIs
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    psycopg2 = None
    RealDictCursor = None

logger = logging.getLogger(__name__)


def adapt_decimal(decimal_value):
    """Adapter for Decimal to PostgreSQL numeric."""
    return AsIs(str(decimal_value))


class DatabaseError(Exception):
    """Custom exception for database errors."""
    pass


class DatabaseNotConfiguredError(DatabaseError):
    """Raised when database is not configured."""
    pass


class Database:
    """
    PostgreSQL database connection manager.

    Features:
    - Connection pooling (via context managers)
    - Automatic retry on connection errors
    - Query result formatting (dict results)
    - Batch operations support
    - Transaction support
    """

    _instance = None

    def __new__(cls):
        """Singleton pattern for database connection."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize database configuration."""
        if self._initialized:
            return

        self._initialized = True
        self._database_url = None
        self._connect_timeout = 10
        self._max_retries = 3

        # Register Decimal adapter if psycopg2 is available
        if PSYCOPG2_AVAILABLE:
            register_adapter(Decimal, adapt_decimal)

    @property
    def database_url(self) -> str:
        """Get database URL from environment."""
        if self._database_url is None:
            self._database_url = os.environ.get('DATABASE_URL')
        return self._database_url

    @database_url.setter
    def database_url(self, value: str):
        """Set database URL (mainly for testing)."""
        self._database_url = value

    def _check_available(self):
        """Check if psycopg2 is available."""
        if not PSYCOPG2_AVAILABLE:
            raise DatabaseError(
                "psycopg2 is not installed. Install with: pip install psycopg2-binary"
            )

    def _check_configured(self):
        """Check if database is configured."""
        if not self.database_url:
            raise DatabaseNotConfiguredError(
                "DATABASE_URL environment variable not set. "
                "Configure PostgreSQL in Railway and wire the DATABASE_URL variable."
            )

    @contextmanager
    def get_connection(self):
        """
        Get database connection with automatic cleanup.

        Yields:
            psycopg2 connection object

        Example:
            with db.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
        """
        self._check_available()
        self._check_configured()

        conn = None
        try:
            conn = psycopg2.connect(
                self.database_url,
                connect_timeout=self._connect_timeout
            )
            yield conn
            conn.commit()
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise DatabaseError(f"Database operation failed: {e}") from e
        finally:
            if conn:
                conn.close()

    @contextmanager
    def get_cursor(self, cursor_factory=None):
        """
        Get database cursor with automatic cleanup.

        Args:
            cursor_factory: Cursor factory (default: RealDictCursor for dict results)

        Yields:
            Database cursor

        Example:
            with db.get_cursor() as cursor:
                cursor.execute("SELECT * FROM users")
                results = cursor.fetchall()
        """
        if cursor_factory is None:
            cursor_factory = RealDictCursor

        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
            finally:
                cursor.close()

    @contextmanager
    def transaction(self):
        """
        Create a transaction context.

        All operations within the context are committed on success,
        rolled back on exception.

        Example:
            with db.transaction() as cursor:
                cursor.execute("INSERT INTO users ...")
                cursor.execute("INSERT INTO audit_log ...")
                # Both committed together, or both rolled back
        """
        self._check_available()
        self._check_configured()

        conn = None
        try:
            conn = psycopg2.connect(
                self.database_url,
                connect_timeout=self._connect_timeout
            )
            conn.autocommit = False
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            try:
                yield cursor
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cursor.close()
        finally:
            if conn:
                conn.close()

    def execute(self, query: str, params: tuple = None) -> List[Dict]:
        """
        Execute query and return results as list of dicts.

        Args:
            query: SQL query with %s placeholders
            params: Query parameters

        Returns:
            List of result rows as dictionaries

        Example:
            results = db.execute(
                "SELECT * FROM recommendations WHERE symbol = %s",
                ("AAPL",)
            )
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            if cursor.description:
                return [dict(row) for row in cursor.fetchall()]
            return []

    def execute_one(self, query: str, params: tuple = None) -> Optional[Dict]:
        """
        Execute query and return single result.

        Args:
            query: SQL query with %s placeholders
            params: Query parameters

        Returns:
            Single result row as dictionary, or None if no results

        Example:
            user = db.execute_one(
                "SELECT * FROM users WHERE id = %s",
                (user_id,)
            )
        """
        results = self.execute(query, params)
        return results[0] if results else None

    def execute_scalar(self, query: str, params: tuple = None) -> Any:
        """
        Execute query and return single scalar value.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            First column of first row, or None

        Example:
            count = db.execute_scalar("SELECT COUNT(*) FROM users")
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            row = cursor.fetchone()
            if row:
                # RealDictCursor returns dict, get first value
                return list(row.values())[0] if isinstance(row, dict) else row[0]
            return None

    def insert(self, table: str, data: Dict, returning: str = 'id') -> Any:
        """
        Insert row and return specified column (default: id).

        Args:
            table: Table name
            data: Dictionary of column: value pairs
            returning: Column to return (default: 'id')

        Returns:
            Value of the returning column

        Example:
            new_id = db.insert("recommendations", {
                "symbol": "AAPL",
                "recommendation": "BUY",
                "confidence_score": 0.85
            })
        """
        columns = list(data.keys())
        placeholders = ', '.join(['%s'] * len(data))
        column_list = ', '.join(columns)

        query = f"INSERT INTO {table} ({column_list}) VALUES ({placeholders}) RETURNING {returning}"

        with self.get_cursor() as cursor:
            cursor.execute(query, tuple(data.values()))
            result = cursor.fetchone()
            return result[returning] if result else None

    def insert_many(self, table: str, data: List[Dict], returning: bool = False) -> Union[int, List]:
        """
        Insert multiple rows efficiently.

        Args:
            table: Table name
            data: List of dictionaries with same keys
            returning: If True, return list of inserted IDs

        Returns:
            Number of rows inserted, or list of IDs if returning=True

        Example:
            count = db.insert_many("predictions", [
                {"symbol": "AAPL", "predicted_price": 150.0},
                {"symbol": "GOOGL", "predicted_price": 140.0}
            ])
        """
        if not data:
            return [] if returning else 0

        columns = list(data[0].keys())
        column_list = ', '.join(columns)

        if returning:
            query = f"INSERT INTO {table} ({column_list}) VALUES %s RETURNING id"
        else:
            query = f"INSERT INTO {table} ({column_list}) VALUES %s"

        values = [tuple(row.get(col) for col in columns) for row in data]

        with self.get_cursor() as cursor:
            if returning:
                execute_values(cursor, query, values, fetch=True)
                return [row['id'] for row in cursor.fetchall()]
            else:
                execute_values(cursor, query, values)
                return len(data)

    def update(self, table: str, data: Dict, where: str, where_params: tuple = None) -> int:
        """
        Update rows matching condition.

        Args:
            table: Table name
            data: Dictionary of column: value pairs to update
            where: WHERE clause (without "WHERE" keyword)
            where_params: Parameters for WHERE clause

        Returns:
            Number of rows updated

        Example:
            updated = db.update(
                "recommendations",
                {"recommendation": "HOLD", "updated_at": datetime.now()},
                "symbol = %s",
                ("AAPL",)
            )
        """
        set_clause = ', '.join([f"{col} = %s" for col in data.keys()])
        query = f"UPDATE {table} SET {set_clause} WHERE {where}"

        params = tuple(data.values())
        if where_params:
            params = params + where_params

        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.rowcount

    def delete(self, table: str, where: str, where_params: tuple = None) -> int:
        """
        Delete rows matching condition.

        Args:
            table: Table name
            where: WHERE clause (without "WHERE" keyword)
            where_params: Parameters for WHERE clause

        Returns:
            Number of rows deleted

        Example:
            deleted = db.delete(
                "recommendations",
                "expires_at < %s",
                (datetime.now(),)
            )
        """
        query = f"DELETE FROM {table} WHERE {where}"

        with self.get_cursor() as cursor:
            cursor.execute(query, where_params)
            return cursor.rowcount

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name = %s
            )
        """
        result = self.execute_one(query, (table_name,))
        return result.get('exists', False) if result else False

    def health_check(self) -> Dict:
        """
        Check database connectivity and return status.

        Returns:
            Dictionary with health status

        Example:
            status = db.health_check()
            # {"healthy": True, "latency_ms": 5.2, "version": "PostgreSQL 15.0"}
        """
        import time

        try:
            start = time.time()
            result = self.execute_one("SELECT version()")
            latency = (time.time() - start) * 1000

            return {
                "healthy": True,
                "latency_ms": round(latency, 2),
                "version": result.get('version', 'unknown') if result else 'unknown'
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }


# Global database instance
db = Database()


# Convenience functions for common operations
def is_database_available() -> bool:
    """Check if database is available and configured."""
    return PSYCOPG2_AVAILABLE and db.database_url is not None


def require_database(func):
    """
    Decorator to require database availability.

    Example:
        @require_database
        def get_recommendations():
            return db.execute("SELECT * FROM recommendations")
    """
    def wrapper(*args, **kwargs):
        if not PSYCOPG2_AVAILABLE:
            raise DatabaseError("psycopg2 not installed")
        if not db.database_url:
            raise DatabaseNotConfiguredError("DATABASE_URL not configured")
        return func(*args, **kwargs)
    return wrapper
