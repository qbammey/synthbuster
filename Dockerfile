# ---- Base image ----
FROM registry.ipol.im/ipol:v2-py3.11

# ---- Install uv ----
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ---- Runtime env ----
ENV HOME=/home/ipol \
    bin=/workdir/bin \
    PYTHONDONTWRITEBYTECODE=1 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PROJECT_ENVIRONMENT=/home/ipol/.uv-envs/dr \
    UV_CACHE_DIR=/home/ipol/.uv-cache

# Create runtime user & dirs; make /opt/uv world-readable
RUN groupadd -g 1000 ipol \
 && useradd -m -u 1000 -g 1000 -d "$HOME" ipol \
 && mkdir -p /workdir "$bin" /opt/uv/python /home/ipol/.uv-envs "$UV_CACHE_DIR" \
 && chown -R ipol:ipol /workdir "$HOME" /opt/uv /home/ipol/.uv-envs "$UV_CACHE_DIR" \
 && chmod -R 755 /opt/uv

# ---- Switch to runtime user before copying/syncing ----
USER ipol
WORKDIR $bin

# Copy project; ensure dir ownership is correct
COPY --chown=ipol:ipol . .
RUN chown -R ipol:ipol /workdir /workdir/bin

# ---- Build step: disable uv cache to avoid stale/permissioned cache ----
ENV UV_NO_CACHE=1
# use a throwaway cache dir for the build layer
ENV UV_CACHE_DIR=/tmp/uv-build-cache
RUN rm -rf "$UV_CACHE_DIR" && mkdir -p "$UV_CACHE_DIR"

# Create the environment OUTSIDE the project folder and install deps
RUN uv sync

# ---- Restore runtime cache settings (re-enable cache) ----
ENV UV_NO_CACHE=false
ENV UV_CACHE_DIR=/home/ipol/.uv-cache

# QoL
ENV PATH=$bin:$PATH

