import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from app.core.config import settings

local_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../oncogemma_local.db"))
sqlite_url = f"sqlite:///{local_db_path}"

db_url = settings.DATABASE_URL

if db_url.startswith("sqlite"):
    engine = create_engine(
        sqlite_url,
        connect_args={"check_same_thread": False, "timeout": 30}
    )
else:
    try:
        engine = create_engine(
            db_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
            connect_args={"connect_timeout": 5}
        )
        with engine.connect() as conn:
            pass
    except Exception as e:
        print(f"[DB Core Warning] Postgres connection failed ({e}). Falling back to shared local SQLite database at {sqlite_url}.")
        engine = create_engine(
            sqlite_url,
            connect_args={"check_same_thread": False, "timeout": 30}
        )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
