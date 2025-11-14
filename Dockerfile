# syntax=docker/dockerfile:1

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libwebsockets-dev \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

# Configure and build
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build --config Release

# Default command runs the built client
CMD ["./build/showdown_client"]