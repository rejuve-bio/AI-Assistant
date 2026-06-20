FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

RUN groupadd --gid 1000 sandbox \
    && useradd --uid 1000 --gid 1000 --create-home --home-dir /home/user --shell /usr/sbin/nologin sandbox \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        coreutils \
        procps \
        r-base \
        samtools \
        bcftools \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir \
        "pandas>=2.2" \
        "numpy>=1.26" \
        "matplotlib>=3.9" \
        "seaborn>=0.13" \
        "scikit-learn>=1.5" \
        "scipy>=1.13" \
        requests \
        networkx \
        statsmodels \
        gseapy \
        anndata \
        scanpy \
        pydeseq2 \
        pyarrow \
        pyliftover \
        feedparser \
        scholarly \
        beautifulsoup4 \
        primer3-py \
        rdkit \
        biopython \
        pysam \
    && mkdir -p /home/user/input /home/user/output \
    && chown -R sandbox:sandbox /home/user

USER sandbox
WORKDIR /home/user

CMD ["python", "--version"]
