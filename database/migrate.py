
"""Initial database bootstrap for local development."""

from database.connection import check_database_connection, engine
from database.migrations import upgrade


def main():
    try:
        check_database_connection()
        upgrade()
        print("PostgreSQL connection successful.")
        print("SQLAlchemy tables created or already present.")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()


# python -m database.migrate