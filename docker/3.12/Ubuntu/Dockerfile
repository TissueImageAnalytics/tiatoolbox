FROM ubuntu:22.04 AS builder-image

# To avoid tzdata blocking the build with frontend questions
ENV DEBIAN_FRONTEND=noninteractive

# Install python3.12
RUN apt-get update && \
    apt install software-properties-common -y &&\
    add-apt-repository ppa:deadsnakes/ppa -y && apt update &&\
    apt-get install -y --no-install-recommends python3.12-venv &&\
    apt-get install libpython3.12-de -y &&\
    apt-get install python3.12-dev -y &&\
    apt-get install build-essential -y &&\
    apt-get clean

# Add env to PATH
RUN python3.12 -m venv /venv
ENV PATH=/venv/bin:$PATH

# install TIAToolbox and its requirements
RUN apt-get update && apt-get install --no-install-recommends -y \
    libopenjp2-7-dev libopenjp2-tools \
    openslide-tools \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir tiatoolbox

# activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
