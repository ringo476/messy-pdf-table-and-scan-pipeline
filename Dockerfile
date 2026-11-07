FROM python:3.12-slim

# Native libraries needed by OpenCV.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# A slow connection shouldn't fail a build that is mostly large downloads. These settings are environment variables,
# so they also reach the helper pip that pip starts to build the package.
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=8

WORKDIR /app
COPY pyproject.toml requirements.lock ./

# Step 1: the dependencies, in a layer of their own. This layer is rebuilt only when pyproject.toml or the lock file
# changes, so an ordinary code change doesn't download hundreds of megabytes again. The empty package lets pip
# install the dependencies without the real code. requirements.lock pins exact versions, so every build installs the
# same packages.
RUN mkdir -p src/pdfpipe && touch src/pdfpipe/__init__.py \
    && pip install --no-cache-dir -c requirements.lock ".[service,llm]" \
    && pip uninstall -y pdfpipe

# Step 2: the model files, downloaded during the build so nothing is fetched while the app is running. This layer is
# also kept when the code changes.
RUN python -c "from rapidocr import RapidOCR; \
from rapid_layout import ModelType, RapidLayout, RapidLayoutInput; \
from rapid_table import RapidTable, RapidTableInput; \
RapidOCR(); \
RapidLayout(RapidLayoutInput(model_type=ModelType.PP_DOC_LAYOUTV3)); \
RapidTable(RapidTableInput(use_ocr=False))"

# Step 3: the real code. Only this and what follows runs again after a code change.
COPY src ./src
# The build folder from step 1 holds a copy of the empty placeholder package. Without removing it, setuptools sees
# that copy as newer than the real files and installs the empty one.
RUN rm -rf build src/pdfpipe.egg-info \
    && pip install --no-cache-dir --no-deps --force-reinstall .

# Check that every model loads, so a broken build fails here and not on the first document.
RUN python -c "from pdfpipe.config import Settings; from pdfpipe.engines import Engines; Engines(Settings(_env_file=None)).warm_up()"

# Run as a normal user, with write access only to the data folder.
RUN useradd --create-home --uid 1000 pdfpipe && mkdir -p /data && chown pdfpipe:pdfpipe /data
USER pdfpipe
ENV PDFPIPE_DATA_DIR=/data
