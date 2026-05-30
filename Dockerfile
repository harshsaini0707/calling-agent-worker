# syntax=docker/dockerfile:1
# Production image for LiveKit Cloud agent hosting.
# Multi-stage uv build: deps + models baked in the build stage, slim non-root
# runtime stage. LiveKit Cloud handles health, scaling, and graceful drain —
# there is no in-process health/metrics server here.

ARG PYTHON_VERSION=3.12
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS base

ENV PYTHONUNBUFFERED=1
# Compile bytecode at install time to reduce cold-start cost
ENV UV_COMPILE_BYTECODE=1

# ── Build stage ───────────────────────────────────────────────────────────────
FROM base AS build

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    ffmpeg \
    libsndfile1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for layer-cache efficiency — code changes won't bust this layer.
# --no-install-project: this is a flat app (agent.py), not a packaged library;
# we run it directly, so there's no build backend / package to install.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# Pre-download ML models (Silero VAD) before copying source so this layer stays
# cached across code-only deploys.
ENV HF_HOME=/app/models
RUN mkdir -p /app/models && uv run --no-sync --module livekit.agents download-files

# Copy remaining source
COPY . .

# ── Production stage ──────────────────────────────────────────────────────────
FROM base AS production

# Runtime-only system deps — no build tools in final image
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
  && rm -rf /var/lib/apt/lists/*

# Non-root user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

WORKDIR /app

# Copy built app + venv from build stage in one layer
COPY --from=build --chown=appuser:appuser /app /app

USER appuser

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models

# agent_name "outbound-caller" is set on the @server.rtc_session decorator;
# the backend dispatch uses that name to find this worker.
CMD ["uv", "run", "--no-sync", "agent.py", "start"]
