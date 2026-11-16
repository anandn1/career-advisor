# --- Python environment and the embedding model -------------------------------------
FROM python:3.12-slim-trixie AS build
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY app ./app
# Download the pinned model now and check its fingerprints, so the app starts offline.
RUN .venv/bin/python -c "from pathlib import Path; \
from app.core.embeddings import fetch_model_files; \
fetch_model_files(Path('/opt/embedding-model'))" \
 && rm -rf /opt/embedding-model/.cache

# --- A few TeX packages from Debian's large bundles -----------------------------------
# texlive-fonts-extra alone is 1.7 GB; the templates need three small packages from it,
# plus biblatex (with biblatex-ext and logreq) and soul. Take just those directories from
# Debian's signed packages.
FROM python:3.12-slim-trixie AS tex-extras
WORKDIR /tmp/debs
RUN apt-get update \
 && apt-get download texlive-fonts-extra texlive-bibtex-extra texlive-plain-generic \
 && for deb in *.deb; do dpkg-deb -x "$deb" unpacked; done \
 && cd unpacked/usr/share/texlive/texmf-dist \
 && find . -type d \( -name fontawesome5 -o -name fontawesome -o -name cfr-lm \
      -o -name biblatex -o -name biblatex-ext -o -name logreq -o -name soul \) \
      -not -path "./doc/*" -prune \
      -exec sh -c 'mkdir -p "/texmf/$1" && cp -r "$1/." "/texmf/$1/"' _ {} \;

# --- Runtime ---------------------------------------------------------------------------
FROM python:3.12-slim-trixie
RUN apt-get update && apt-get install -y --no-install-recommends \
      texlive-latex-base texlive-latex-recommended texlive-latex-extra \
      texlive-fonts-recommended lmodern cm-super \
 && rm -rf /var/lib/apt/lists/*
COPY --from=tex-extras /texmf /usr/local/share/texmf
RUN mktexlsr \
 && for map in fontawesome5 fontawesome clm; do updmap-sys --enable Map=$map.map; done

RUN useradd --create-home --uid 1000 app && mkdir /data && chown app /data
WORKDIR /app
COPY --from=build /app/.venv ./.venv
COPY --from=build /opt/embedding-model /opt/embedding-model
COPY alembic.ini docker-entrypoint.sh ./
COPY migrations ./migrations
COPY templates ./templates
COPY app ./app

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    EMBEDDING_MODEL_DIR=/opt/embedding-model \
    HF_HUB_OFFLINE=1 \
    ANONYMIZED_TELEMETRY=False
USER app
EXPOSE 8000
HEALTHCHECK --start-period=15m --start-interval=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"
ENTRYPOINT ["./docker-entrypoint.sh"]
