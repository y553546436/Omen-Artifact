FROM nvidia/cuda:12.6.2-runtime-ubuntu24.04

RUN apt-get update && \
    apt-get install -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.10 python3.10-venv && \
    rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /omen/venv

ENV PATH="/omen/venv/bin:$PATH"

WORKDIR /omen

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
