import os
import sys
import uuid
from datetime import datetime, timezone

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

CLOUD_DB_URL = os.environ.get("DATABASE_URL")
if not CLOUD_DB_URL:
    print("Error: DATABASE_URL environment variable is required.")
    sys.exit(1)

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.models.case import Case
from app.models.stage_execution import StageExecution
from worker.report import run_report

CASE_ID = "9d16e702-68fc-4ff9-992a-242a4768ab60"
CASE_UUID = uuid.UUID(CASE_ID)

def main():
    print(f"\nGenerating Stage 6 CAP Report for Case {CASE_ID}...")
    engine = create_engine(CLOUD_DB_URL)
    Session = sessionmaker(bind=engine)
    db = Session()

    stmt_report_exec = select(StageExecution).where(
        StageExecution.case_id == CASE_UUID,
        StageExecution.stage == "report"
    ).order_by(StageExecution.attempt.desc())
    report_exec = db.scalars(stmt_report_exec).first()
    
    if not report_exec:
        report_exec = StageExecution(
            id=uuid.uuid4(),
            case_id=CASE_UUID,
            stage="report",
            attempt=1,
            status="running",
            input_ref={"case_id": CASE_ID},
            started_at=datetime.now(timezone.utc)
        )
        db.add(report_exec)
        db.commit()
    else:
        report_exec.status = "running"
        report_exec.error = None
        db.commit()

    report_uri, report_meta = run_report(report_exec, db)
    print(f"Report successfully generated!\nReport URI: {report_uri}\nMetadata: {report_meta}")
    db.close()

if __name__ == "__main__":
    main()
