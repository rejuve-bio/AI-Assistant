import logging
import os
from apscheduler.schedulers.background import BackgroundScheduler
from app.ingestion.pubmed_ingestion import PubMedIngester

logger = logging.getLogger(__name__)

# Default: Run at 2 AM every day
SCHEDULE_CRON = os.getenv("PUBMED_INGESTION_SCHEDULE", "0 2 * * *")
ENABLED = os.getenv("PUBMED_INGESTION_ENABLED", "true").lower() == "true"

def _run_ingestion_job(qdrant_client, mongo_db):
    try:
        logger.info("[scheduler] Starting scheduled PubMed ingestion cycle...")
        ingester = PubMedIngester(qdrant_client, mongo_db)
        ingester.ingest()
        logger.info("[scheduler] Scheduled PubMed ingestion cycle finished.")
    except Exception as exc:
        logger.error(f"[scheduler] Scheduled PubMed ingestion cycle failed: {exc}")


def start_scheduler(app):
    if not ENABLED or app.testing:
        logger.info("PubMed ingestion scheduler is disabled or in testing mode.")
        return None

    try:
        parts = SCHEDULE_CRON.split()
        if len(parts) != 5:
            logger.warning(f"Invalid cron format '{SCHEDULE_CRON}', defaulting to '0 2 * * *'")
            minute, hour, day, month, day_of_week = "0", "2", "*", "*", "*"
        else:
            minute, hour, day, month, day_of_week = parts

        scheduler = BackgroundScheduler(daemon=True)
        scheduler.add_job(
            _run_ingestion_job,
            "cron",
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            day_of_week=day_of_week,
            args=[app.qdrant_client, app.mongo_db], 
            id="pubmed_ingestion_job",
            replace_existing=True,
        )
        scheduler.start()
        logger.info(f"Started PubMed ingestion scheduler (cron: {SCHEDULE_CRON})")
        return scheduler
    except Exception as exc:
        logger.error(f"Failed to start PubMed ingestion scheduler: {exc}")
        return None
