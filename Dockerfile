FROM python:3.10-slim AS assistant-base

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /AI-Assistant

RUN mkdir -p /AI-Assistant/logfiles \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ncbi-blast+ \
        wget \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU is installed separately to use its smaller CPU-only index.
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml poetry.lock* ./
RUN pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-root --only main


# Production bioinformatics target with R, Bioconductor, and PLINK2.
FROM assistant-base AS assistant-full

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libbz2-dev \
        libcurl4-openssl-dev \
        libhdf5-dev \
        liblzma-dev \
        libssl-dev \
        libxml2-dev \
        r-base \
        r-base-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN Rscript -e "\
    options(repos = c(CRAN = 'https://cloud.r-project.org'), timeout = 1200); \
    install.packages('BiocManager', Ncpus = 2); \
    BiocManager::install( \
        c('impute', 'preprocessCore', 'DESeq2', 'limma', 'edgeR', 'Biostrings', 'GenomicRanges'), \
        ask = FALSE, update = FALSE, Ncpus = 2)"

RUN Rscript -e "\
    options(repos = c(CRAN = 'https://cloud.r-project.org'), timeout = 1200); \
    install.packages('fs', Ncpus = 1)"

RUN Rscript -e "\
    options(repos = c(CRAN = 'https://cloud.r-project.org'), timeout = 1200); \
    install.packages( \
        c('ggplot2', 'dplyr', 'tidyr', 'survival', 'WGCNA', 'igraph'), \
        Ncpus = 1)"

# Prefer Arrow's prebuilt C++ libraries instead of an expensive source build.
RUN LIBARROW_BINARY=true NOT_CRAN=true Rscript -e "\
    options(repos = c(CRAN = 'https://cloud.r-project.org'), timeout = 1200); \
    install.packages('arrow', Ncpus = 2)"

RUN Rscript -e "\
    required <- c( \
        'BiocManager', 'ggplot2', 'dplyr', 'tidyr', 'survival', 'WGCNA', \
        'arrow', 'igraph', 'DESeq2', 'limma', 'edgeR', 'Biostrings', \
        'GenomicRanges'); \
    missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]; \
    if (length(missing)) stop('Missing R packages: ', paste(missing, collapse = ', ')); \
    cat('R packages installed and verified\n')"

RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/mf.sh \
    && bash /tmp/mf.sh -b -p /opt/conda \
    && rm /tmp/mf.sh \
    && /opt/conda/bin/conda install -y -c bioconda plink2 \
    && /opt/conda/bin/conda clean -afy \
    && ln -sf /opt/conda/bin/plink2 /usr/local/bin/plink2

COPY . /AI-Assistant

CMD ["gunicorn", "-w", "4", "--bind", "0.0.0.0:$FLASK_PORT", "--timeout", "600", "run:app"]


# Default development target: Python application without the R toolchain.
FROM assistant-base AS assistant-lite

COPY . /AI-Assistant

CMD ["gunicorn", "-w", "4", "--bind", "0.0.0.0:$FLASK_PORT", "--timeout", "600", "run:app"]
