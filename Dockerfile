FROM registry.ipol.im/ipol:v2-py3.11

# uv binaries
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create runtime user and all dirs as root, then chown them
ENV HOME=/home/ipol
RUN groupadd -g 1000 ipol \
 && useradd -m -u 1000 -g 1000 -d "$HOME" ipol \
 && mkdir -p /workdir /workdir/bin /home/ipol/.uv-envs \
 && chown -R ipol:ipol /workdir "$HOME" /home/ipol/.uv-envs

# Switch to runtime user before touching the project
USER ipol

# Project root
ENV bin=/workdir/bin
WORKDIR $bin

# Copy code (files come in as ipol)
COPY --chown=ipol:ipol . .

# -------- uv: make it permission-proof --------
# 1) Don't use any uv cache (prevents stale root-owned cache reads)
ENV UV_NO_CACHE=1
# 2) Force system Python from the base image (avoid uv-managed interpreters)
ENV UV_PYTHON=/usr/local/bin/python3.11
# 3) Keep the environment OUTSIDE the project tree
ENV UV_PROJECT_ENVIRONMENT=/home/ipol/.uv-envs/dr

# Belt & braces: remove any stray caches that might exist in base layers
RUN rm -rf /home/ipol/.uv-cache /root/.cache/uv /tmp/uv-cache || true

# Create the environment and install deps as ipol
RUN uv sync

# QoL env
ENV PYTHONDONTWRITEBYTECODE=1 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    PATH=$bin:$PATH

