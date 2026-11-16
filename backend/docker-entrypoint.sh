#!/bin/sh
# Prepare the databases, then start the API. Each step skips work that is already done,
# so only the first start is slow (the interview store takes a few minutes to embed).
set -e

echo "Upgrading database tables..."
alembic upgrade head

echo "Loading resume templates..."
python -m app.db.seed_templates

echo "Building the interview question store if it is missing..."
python -m app.agents.ingestion.seed_interview_db --if-empty

# One process only: at startup the API marks interrupted runs as failed.
exec uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
