#!/usr/bin/env python3
"""
PostgreSQL Schema Migration for Railway Deployment.

This script initializes the database schema for the Stock Analytics Engine.
Run this after adding PostgreSQL to your Railway project.

Usage:
    # Set DATABASE_URL environment variable, then run:
    python migrate_postgres.py

    # Or pass DATABASE_URL as argument:
    python migrate_postgres.py "postgresql://user:pass@host:port/db"
"""

import os
import sys
import argparse
from pathlib import Path


def get_database_url(args_url: str = None) -> str:
    """Get database URL from args or environment."""
    if args_url:
        return args_url

    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        print("Error: DATABASE_URL not set.")
        print("\nSet it via environment variable:")
        print("  export DATABASE_URL='postgresql://...'")
        print("\nOr pass it as an argument:")
        print("  python migrate_postgres.py 'postgresql://...'")
        sys.exit(1)

    return database_url


def run_migration(database_url: str, schema_path: Path) -> bool:
    """Execute the schema migration."""
    try:
        import psycopg2
    except ImportError:
        print("Error: psycopg2 not installed.")
        print("Install it with: pip install psycopg2-binary")
        sys.exit(1)

    # Read schema file
    if not schema_path.exists():
        print(f"Error: Schema file not found: {schema_path}")
        sys.exit(1)

    print(f"Reading schema from: {schema_path}")
    schema = schema_path.read_text()

    # Connect and execute
    print(f"Connecting to database...")
    conn = None
    try:
        conn = psycopg2.connect(database_url)
        conn.autocommit = False

        with conn.cursor() as cursor:
            print("Executing schema migration...")
            cursor.execute(schema)

        conn.commit()
        print("\n" + "=" * 50)
        print("Migration completed successfully!")
        print("=" * 50)

        # Verify tables created
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """)
            tables = cursor.fetchall()

            print(f"\nCreated {len(tables)} tables:")
            for table in tables:
                print(f"  - {table[0]}")

            # Check views
            cursor.execute("""
                SELECT table_name
                FROM information_schema.views
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            views = cursor.fetchall()

            print(f"\nCreated {len(views)} views:")
            for view in views:
                print(f"  - {view[0]}")

            # Check functions
            cursor.execute("""
                SELECT routine_name
                FROM information_schema.routines
                WHERE routine_schema = 'public'
                  AND routine_type = 'FUNCTION'
                ORDER BY routine_name
            """)
            functions = cursor.fetchall()

            print(f"\nCreated {len(functions)} functions:")
            for func in functions:
                print(f"  - {func[0]}")

        return True

    except psycopg2.Error as e:
        print(f"\nDatabase error: {e}")
        if conn:
            conn.rollback()
        return False

    except Exception as e:
        print(f"\nUnexpected error: {e}")
        if conn:
            conn.rollback()
        return False

    finally:
        if conn:
            conn.close()


def verify_connection(database_url: str) -> bool:
    """Test database connection before migration."""
    try:
        import psycopg2
        conn = psycopg2.connect(database_url, connect_timeout=10)
        conn.close()
        print("Database connection verified.")
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run PostgreSQL schema migration for Stock Analytics Engine"
    )
    parser.add_argument(
        'database_url',
        nargs='?',
        help='PostgreSQL connection URL (or set DATABASE_URL env var)'
    )
    parser.add_argument(
        '--schema',
        type=Path,
        default=None,
        help='Path to schema.sql file'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify connection, do not run migration'
    )

    args = parser.parse_args()

    # Get database URL
    database_url = get_database_url(args.database_url)

    # Mask password for display
    display_url = database_url
    if '@' in display_url:
        parts = display_url.split('@')
        creds = parts[0].rsplit(':', 1)
        if len(creds) > 1:
            display_url = f"{creds[0]}:****@{parts[1]}"
    print(f"Database: {display_url}")

    # Verify connection
    if not verify_connection(database_url):
        sys.exit(1)

    if args.verify_only:
        print("Connection verified. Exiting (--verify-only mode).")
        sys.exit(0)

    # Determine schema path
    if args.schema:
        schema_path = args.schema
    else:
        # Look for schema.sql relative to this script
        script_dir = Path(__file__).parent
        schema_path = script_dir.parent / 'schema.sql'

    # Run migration
    success = run_migration(database_url, schema_path)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
