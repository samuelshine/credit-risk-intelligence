# syntax=docker/dockerfile:1
# Credit Risk Intelligence Platform
#
# Multi-stage build: the `builder` stage compiles wheels into a venv, the
# final stage copies only that venv plus the application code, so the shipped
# image carries no compiler toolchain. One image serves both roles the
# compose file needs - `etl` (python -m src.data.loader) and `api` (uvicorn)
# - selected by the command each service passes at run time, not by building
# two separate images for what is otherwise identical code and dependencies.

FROM python:3.12-slim AS builder

# build-essential: LightGBM's sdist path and a couple of scientific-stack
# dependencies compile C extensions if no matching manylinux wheel exists for
# this exact platform/Python combination - cheap insurance against a build
# that works today and breaks on the next patch release.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


FROM python:3.12-slim AS runtime

# libgomp1: LightGBM's Linux wheel links against GNU OpenMP at runtime but
# does not bundle it - without this the import fails with a bare
# "libgomp.so.1: cannot open shared object file" the first time a model is
# trained or loaded, which is a confusing failure to debug blind, so it is
# documented here rather than left to be rediscovered.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root: nothing this process does needs root, and running as root in a
# container an evaluator pulls and runs is an avoidable risk for no benefit.
RUN groupadd --gid 1000 app && useradd --uid 1000 --gid app --create-home app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --chown=app:app . .

# data/ and models/ are populated at runtime (ETL output, trained artifacts)
# and are mounted as volumes by docker-compose - created here with the right
# ownership so a first run doesn't hit a root-owned directory from the image
# layer when the volume is a fresh bind mount.
RUN mkdir -p /app/data /app/models && chown -R app:app /app/data /app/models

USER app

EXPOSE 8000

# No CMD: docker-compose.yml gives each service (etl, api) its own command,
# since they share this one image but do different jobs.
