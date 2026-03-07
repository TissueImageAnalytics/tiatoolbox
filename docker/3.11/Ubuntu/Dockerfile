FROM ubuntu:24.04 AS builder-image

# To avoid tzdata blocking the build with frontend questions
ENV DEBIAN_FRONTEND=noninteractive

# Install python3.11
RUN apt-get update && \
    apt install software-properties-common -y &&\
    add-apt-repository ppa:deadsnakes/ppa -y && apt update &&\
    apt-get install -y --no-install-recommends python3.11-venv &&\
    apt-get install libpython3.11-dev -y &&\
    apt-get install python3.11-dev -y &&\
    apt-get install build-essential -y &&\
    apt-get clean

# Add env to PATH
RUN python3.11 -m venv /venv
ENV PATH=/venv/bin:$PATH

# install TIAToolbox and its requirements
RUN apt-get update && apt-get install --no-install-recommends -y \
    libopenjp2-7-dev libopenjp2-tools \
    sqlite3 libsqlite3-0 \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir tiatoolbox

# activate virtual environment
ENV VIRTUAL_ENV=/venv
ENV PATH="/venv/bin:$PATH"
