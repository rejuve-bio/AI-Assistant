FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /AI-Assistant
RUN mkdir -p /AI-Assistant/logfiles

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    libhdf5-dev zlib1g-dev libbz2-dev liblzma-dev \
    # R base
    r-base r-base-dev \
    # BLAST+
    ncbi-blast+ \
    && rm -rf /var/lib/apt/lists/*

# ── R packages — CRAN + Bioconductor ─────────────────────────────────────────
RUN Rscript -e "\
    options(repos = c(CRAN = 'https://cloud.r-project.org')); \
    install.packages(c('BiocManager', 'ggplot2', 'dplyr', 'tidyr', 'survival', 'WGCNA', 'arrow', 'igraph'), \
                     Ncpus = 4, quiet = TRUE); \
    BiocManager::install(c('DESeq2', 'limma', 'edgeR', 'Biostrings', 'GenomicRanges'), \
                         ask = FALSE, update = FALSE, Ncpus = 4); \
    cat('R packages installed\n')"

# ── PLINK2 via miniforge/bioconda ─────────────────────────────────────────────
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/mf.sh \
    && bash /tmp/mf.sh -b -p /opt/conda \
    && rm /tmp/mf.sh \
    && /opt/conda/bin/conda install -y -c bioconda plink2 \
    && /opt/conda/bin/conda clean -afy \
    && ln -sf /opt/conda/bin/plink2 /usr/local/bin/plink2

# ── PyTorch CPU (installed separately to use correct index) ───────────────────
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# ── All other Python dependencies ─────────────────────────────────────────────
COPY pyproject.toml poetry.lock* ./
RUN pip install poetry && poetry config virtualenvs.create false && poetry install --no-root --only main

# ── Application code ──────────────────────────────────────────────────────────
COPY . /AI-Assistant

CMD ["gunicorn", "-w", "4", "--bind", "0.0.0.0:$FLASK_PORT", "--timeout", "600", "run:app"]
