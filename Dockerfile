# ---- Base image with Python 3.11 ----
FROM registry.ipol.im/ipol:v2-py3.11

# ---- Install uv ----
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ---- Environment (runtime defaults) ----
ENV HOME=/home/ipol \
    bin=/workdir/bin \
    PYTHONDONTWRITEBYTECODE=1 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PROJECT_ENVIRONMENT=/home/ipol/.uv-envs/dr \
    UV_CACHE_DIR=/home/ipol/.uv-cache

# Create runtime user and directories, set ownership/permissions
RUN groupadd -g 1000 ipol \
 && useradd -m -u 1000 -g 1000 -d "$HOME" ipol \
 && mkdir -p /workdir "$bin" /opt/uv/python /home/ipol/.uv-envs "$UV_CACHE_DIR" \
 && chown -R ipol:ipol /workdir "$HOME" /opt/uv /home/ipol/.uv-envs "$UV_CACHE_DIR" \
 && chmod -R 755 /opt/uv

# ---- Switch to runtime user before copying code / syncing ----
USER ipol
WORKDIR $bin

# Copy project files; ensure directory itself is owned by ipol
COPY --chown=ipol:ipol . .
RUN chown -R ipol:ipol /workdir /workdir/bin

# ---- Build step: disable uv cache to avoid stale/permissioned cache paths ----
# Use a throwaway cache dir under /tmp and bypass caching entirely for the build.
ENV UV_NO_CACHE=1
ENV UV_CACHE_DIR=/tmp/uv-build-cache
RUN rm -rf "$UV_CACHE_DIR" && mkdir -p "$UV_CACHE_DIR"

# Create the environment OUTSIDE the project folder and install deps
RUN uv sync

# Restore runtime cache location (optional; uv will recreate it at runtime)
ENV UV_NO_CACHE= \
    UV_CACHE_DIR=/home/ipol/.uv-cache

# QoL: put project bin on PATH
ENV PATH=$bin:$PATH

