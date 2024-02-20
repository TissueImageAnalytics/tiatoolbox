FROM python:3.11-slim-bullseye

#get linux packages
RUN apt-get -y update && apt-get -y install --no-install-recommends \
    libopenjp2-7-dev libopenjp2-tools \
    openslide-tools \
    libgl1 \
    build-essential \
    && pip3 --no-cache-dir install tiatoolbox \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# set the entry point to bash
ENTRYPOINT ["/bin/bash"]
