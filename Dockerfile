# ---- Base image ----
FROM registry.ipol.im/ipol:v2-py3.11

# ---- Install uv ----
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ---- Runtime env (do NOT set UV_* cache vars yet) ----
ENV HOME=/home/ipol \
    bin=/workdir/bin \
    PYTHONDONTWRITEBYTECODE=1 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PROJECT_ENVIRONMENT=/home/ipol/.uv-envs/dr

# Create runtime user & dirs; make /opt/uv world-readable
RUN groupadd -g 1000 ipol \
 && useradd -m -u 1000 -g 1000 -d "$HOME" ipol \
 && mkdir -p /workdir "$bin" /opt/uv/python /home/ipol/.uv-envs \
 && chown -R ipol:ipol /workdir "$HOME" /opt/uv /home/ipol/.uv-envs \
 && chmod -R 755 /opt/uv

# ---- Switch to runtime user before copying/syncing ----
USER ipol
WORKDIR $bin

# Copy project; ensure dir ownership is correct
COPY --chown=ipol:ipol . .
RUN chown -R ipol:ipol /workdir /workdir/bin

# ---- Build step: use a throwaway cache and disable caching ----
ENV UV_NO_CACHE=1
ENV UV_CACHE_DIR=/tmp/uv-build-cache
RUN rm -rf "$UV_CACHE_DIR" && mkdir -p "$UV_CACHE_DIR"

# Create the environment OUTSIDE the project folder and install deps
RUN uv sync

# ---- Prepare clean runtime cache (create as ipol, not root) ----
# Use /home/ipol/.uv-cache at runtime, but recreate it now as ipol.
ENV UV_NO_CACHE=false
ENV UV_CACHE_DIR=/home/ipol/.uv-cache
RUN rm -rf "$UV_CACHE_DIR" && mkdir -p "$UV_CACHE_DIR"

# QoL
ENV PATH=$bin:$PATH

