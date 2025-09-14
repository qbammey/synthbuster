FROM registry.ipol.im/ipol:v2-py3.11

# uv binaries
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create runtime user & dirs first
ENV HOME=/home/ipol
RUN groupadd -g 1000 ipol \
 && useradd -m -u 1000 -g 1000 -d "$HOME" ipol \
 && mkdir -p /workdir /home/ipol/.uv-envs \
 && chown -R ipol:ipol /workdir "$HOME"

# Switch to runtime user before we touch the project
USER ipol

# Project root inside the image
ENV bin=/workdir/bin
WORKDIR $bin

# Copy code and ensure ownership of the directory itself
COPY --chown=ipol:ipol . .
RUN chown -R ipol:ipol /workdir /workdir/bin

# -------- uv settings (make it permission-proof) --------
# 1) Don’t use any cache at build or runtime (prevents stale root-owned cache reads)
ENV UV_NO_CACHE=1
# 2) Use system Python from the base image, avoid uv-managed interpreters entirely
ENV UV_PYTHON=/usr/local/bin/python3.11
# 3) Keep the environment OUTSIDE the project tree
ENV UV_PROJECT_ENVIRONMENT=/home/ipol/.uv-envs/dr

# Nuke any stray caches that might exist in the base image layers (belt & braces)
RUN rm -rf /home/ipol/.uv-cache /root/.cache/uv /tmp/uv-cache || true

# Create the environment and install deps
RUN uv sync

# QoL
ENV PYTHONDONTWRITEBYTECODE=1 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    PATH=$bin:$PATH

