FROM ubuntu:24.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential cmake pkg-config \
  libwebsockets-dev libcurl4-openssl-dev ca-certificates && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

ENV PS_USER="" PS_PASSWORD=""
CMD ["./build/showdown_client"]
