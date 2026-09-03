import logging
import os
from apscheduler.schedulers.background import BackgroundScheduler
from app.ingestion.pubmed_ingestion import PubMedIngester
from app.ingestion.europepmc_ingestion import EuropePMCIngester

logger = logging.getLogger(__name__)

# Default: Run at 2 AM every day
SCHEDULE_CRON = os.getenv("PUBMED_INGESTION_SCHEDULE", "0 2 * * *")
ENABLED = os.getenv("PUBMED_INGESTION_ENABLED", "true").lower() == "true"

# Europe PMC: runs 15 min after PubMed so cross-source dedup catches overlaps.
EPMC_ENABLED = os.getenv("EPMC_INGESTION_ENABLED", "true").lower() == "true"
EPMC_SCHEDULE_CRON = os.getenv("EPMC_INGESTION_SCHEDULE", "15 2 * * *")

# Fine-tuning QA generation: runs after ingestion to create training pairs.
FT_GENERATION_ENABLED = os.getenv("FT_GENERATION_ENABLED", "false").lower() == "true"
FT_SCHEDULE_CRON = os.getenv("FT_GENERATION_SCHEDULE", "30 2 * * *")


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


def _run_qa_generation_job(llm, mongo_db):
    """Generate QA fine-tuning pairs for recently ingested papers."""
    try:
        from app.ingestion.qa_generator import QAGenerator
        logger.info("[scheduler] Starting scheduled QA generation cycle...")
        generator = QAGenerator(llm, mongo_db)
        stats = generator.generate_from_index(source="pubmed")
        logger.info(f"[scheduler] Scheduled QA generation cycle finished: {stats}")
    except Exception as exc:
        logger.error(f"[scheduler] Scheduled QA generation cycle failed: {exc}")


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

        # Optionally schedule QA pair generation after ingestion.
        if FT_GENERATION_ENABLED:
            ft_parts = FT_SCHEDULE_CRON.split()
            if len(ft_parts) != 5:
                ft_parts = ["30", "2", "*", "*", "*"]
            ft_min, ft_hr, ft_day, ft_mon, ft_dow = ft_parts

            scheduler.add_job(
                _run_qa_generation_job,
                "cron",
                minute=ft_min,
                hour=ft_hr,
                day=ft_day,
                month=ft_mon,
                day_of_week=ft_dow,
                args=[app.llm, app.mongo_db],
                id="ft_qa_generation_job",
                replace_existing=True,
            )
            logger.info(f"Started QA generation scheduler (cron: {FT_SCHEDULE_CRON})")

        scheduler.start()
        return scheduler
    except Exception as exc:
        logger.error(f"Failed to start PubMed ingestion scheduler: {exc}")
        return None
