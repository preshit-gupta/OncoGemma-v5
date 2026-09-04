import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from app.core.config import settings

local_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../oncogemma_local.db"))
sqlite_url = f"sqlite:///{local_db_path}"

db_url = settings.DATABASE_URL

# Auto-detect Cloud SQL Unix Socket on Cloud Run
cloudsql_instances = [
    "/cloudsql/oncogemma:us-central1:oncogemma-dev-psql"
]
if os.path.exists("/cloudsql"):
    try:
        for entry in os.listdir("/cloudsql"):
            full_p = f"/cloudsql/{entry}"
            if os.path.isdir(full_p) or ":" in entry:
                cloudsql_instances.append(full_p)
    except Exception:
        pass

for sock in cloudsql_instances:
    if os.path.exists(sock) and ("localhost" in db_url or "127.0.0.1" in db_url):
        cloud_pw = os.getenv("DB_PASSWORD", "oncogemma_secure_cloud_password")
        db_url = f"postgresql+psycopg2://oncogemma:{cloud_pw}@/oncogemma_db?host={sock}"
        print(f"[DB Core] Connected to Cloud SQL via unix socket at {sock}")
        break

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
