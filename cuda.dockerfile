FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        software-properties-common \
        unzip \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /omen/venv

ENV PATH="/omen/venv/bin:$PATH"

WORKDIR /omen

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN rm -rf requirements.txt

WORKDIR /omen/workspace
