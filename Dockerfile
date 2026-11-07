FROM python:3.12-slim

# Native libraries needed by OpenCV.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml requirements.lock ./
COPY src ./src

# requirements.lock pins exact versions, so every build installs the same packages.
RUN pip install --no-cache-dir -c requirements.lock ".[service,llm]"

# Download the model files during the build, so nothing is fetched while the app is running.
RUN python -c "from pdfpipe.config import Settings; from pdfpipe.engines import Engines; Engines(Settings(_env_file=None)).warm_up()"

# Run as a normal user, with write access only to the data folder.
RUN useradd --create-home --uid 1000 pdfpipe && mkdir -p /data && chown pdfpipe:pdfpipe /data
USER pdfpipe
ENV PDFPIPE_DATA_DIR=/data
