FROM debian:bookworm-slim AS builder
ARG API_ARCH=-march=haswell
RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc libc6-dev zlib1g-dev curl ca-certificates make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY Makefile ./
COPY src ./src
COPY resources ./resources

# Download references dataset at build time and build the index.
RUN curl -fsSL -o references.json.gz \
      https://github.com/zanfranceschi/rinha-de-backend-2026/raw/main/resources/references.json.gz \
    && make build_index API_ARCH="$API_ARCH" \
    && ./build_index references.json.gz index.bin \
    && ls -lh index.bin

# --- PGO step 1: build instrumented api, run it against the example payloads
# to collect a profile that reflects the real query distribution. ---
RUN mkdir -p /tmp/pgo \
    && make api API_ARCH="$API_ARCH" PGO=generate \
    && ./api --pgo-train index.bin resources/example-payloads.json \
    && rm -f api

# --- PGO step 2: rebuild api + lb with the collected profile. ---
RUN make api lb API_ARCH="$API_ARCH" PGO=use \
    && ls -lh api lb

FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
      libc6 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /build/api /app/api
COPY --from=builder /build/lb  /app/lb
COPY --from=builder /build/index.bin /app/index.bin
# /sockets is the shared volume mount point for AF_UNIX control sockets.
RUN mkdir -p /sockets
# Default entrypoint runs the API; the LB service overrides command in compose.
CMD ["/app/api", "/app/index.bin", "/sockets/api.sock"]
