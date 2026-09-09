import logging
import os
from apscheduler.schedulers.background import BackgroundScheduler
from app.ingestion.pubmed_ingestion import PubMedIngester
from app.ingestion.europepmc_ingestion import EuropePMCIngester
from app.ingestion.semantic_scholar_ingestion import SemanticScholarIngester

logger = logging.getLogger(__name__)

# Default: Run at 2 AM every day
SCHEDULE_CRON = os.getenv("PUBMED_INGESTION_SCHEDULE", "0 2 * * *")
ENABLED = os.getenv("PUBMED_INGESTION_ENABLED", "true").lower() == "true"

# Europe PMC: runs 15 min after PubMed so cross-source dedup catches overlaps.
EPMC_ENABLED = os.getenv("EPMC_INGESTION_ENABLED", "true").lower() == "true"
EPMC_SCHEDULE_CRON = os.getenv("EPMC_INGESTION_SCHEDULE", "15 2 * * *")

# Semantic Scholar: runs 30 min after PubMed, adds citation-heavy uniquely indexed papers.
S2_ENABLED = os.getenv("S2_INGESTION_ENABLED", "true").lower() == "true"
S2_SCHEDULE_CRON = os.getenv("S2_INGESTION_SCHEDULE", "30 2 * * *")



def _run_ingestion_job(qdrant_client, mongo_db):
    try:
        logger.info("[scheduler] Starting scheduled PubMed ingestion cycle...")
        ingester = PubMedIngester(qdrant_client, mongo_db)
        stats = ingester.ingest()
        logger.info(f"[scheduler] Scheduled PubMed ingestion cycle finished: {stats}")
    except Exception as exc:
        logger.error(f"[scheduler] Scheduled PubMed ingestion cycle failed: {exc}")


def _run_epmc_ingestion_job(qdrant_client, mongo_db):
    """Ingest papers from Europe PMC (catches what PubMed misses)."""
    try:
        logger.info("[scheduler] Starting scheduled Europe PMC ingestion cycle...")
        ingester = EuropePMCIngester(qdrant_client, mongo_db)
        stats = ingester.ingest()
        logger.info(f"[scheduler] Europe PMC ingestion cycle finished: {stats}")
    except Exception as exc:
        logger.error(f"[scheduler] Europe PMC ingestion cycle failed: {exc}")


def _run_s2_ingestion_job(qdrant_client, mongo_db):
    """Ingest papers from Semantic Scholar (adds AI-curated uniqueness)."""
    try:
        logger.info("[scheduler] Starting scheduled Semantic Scholar ingestion cycle...")
        ingester = SemanticScholarIngester(qdrant_client, mongo_db)
        stats = ingester.ingest()
        logger.info(f"[scheduler] Semantic Scholar ingestion cycle finished: {stats}")
    except Exception as exc:
        logger.error(f"[scheduler] Semantic Scholar ingestion cycle failed: {exc}")



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
        logger.info(f"Started PubMed ingestion scheduler (cron: {SCHEDULE_CRON})")

        # Europe PMC ingestion (runs after PubMed, cross-source dedup via DOI).
        if EPMC_ENABLED:
            epmc_parts = EPMC_SCHEDULE_CRON.split()
            if len(epmc_parts) != 5:
                epmc_parts = ["15", "2", "*", "*", "*"]
            epmc_min, epmc_hr, epmc_day, epmc_mon, epmc_dow = epmc_parts

            scheduler.add_job(
                _run_epmc_ingestion_job,
                "cron",
                minute=epmc_min,
                hour=epmc_hr,
                day=epmc_day,
                month=epmc_mon,
                day_of_week=epmc_dow,
                args=[app.qdrant_client, app.mongo_db],
                id="epmc_ingestion_job",
                replace_existing=True,
            )
            logger.info(f"Started Europe PMC ingestion scheduler (cron: {EPMC_SCHEDULE_CRON})")

        # Semantic Scholar ingestion (runs after EPMC).
        if S2_ENABLED:
            s2_parts = S2_SCHEDULE_CRON.split()
            if len(s2_parts) != 5:
                s2_parts = ["30", "2", "*", "*", "*"]
            s2_min, s2_hr, s2_day, s2_mon, s2_dow = s2_parts

            scheduler.add_job(
                _run_s2_ingestion_job,
                "cron",
                minute=s2_min,
                hour=s2_hr,
                day=s2_day,
                month=s2_mon,
                day_of_week=s2_dow,
                args=[app.qdrant_client, app.mongo_db],
                id="s2_ingestion_job",
                replace_existing=True,
            )
            logger.info(f"Started Semantic Scholar ingestion scheduler (cron: {S2_SCHEDULE_CRON})")

        scheduler.start()
        return scheduler
    except Exception as exc:
        logger.error(f"Failed to start PubMed ingestion scheduler: {exc}")
        return None
