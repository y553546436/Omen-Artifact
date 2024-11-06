FROM nvidia/cuda:12.6-runtime-ubuntu24.04

RUN apt-get update && \
    apt-get install -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.11 python3.11-venv && \
    rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /omen/venv

ENV PATH="/omen/venv/bin:$PATH"

WORKDIR /omen

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN rm -rf requirements.txt

WORKDIR /omen/workspace
