FROM python:3.10-slim AS python

WORKDIR /omen

# copy requirements
COPY requirements.txt .

# create virtual environment
RUN python -m venv /omen/venv

# activate virtual environment
ENV PATH="/omen/venv/bin:$PATH"

# install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim-bookworm AS runtime

COPY --from=python /omen/venv /omen/venv

# install build dependencies gcc and g++
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/omen/venv/bin:$PATH"

WORKDIR /omen/workspace
