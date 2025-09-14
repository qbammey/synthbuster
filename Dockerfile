# ---- Base image ----
FROM registry.ipol.im/ipol:v2-py3.11

# ---- Install uv ----
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ---- Common env ----
ENV HOME=/home/ipol \
    bin=/workdir/bin \
    PYTHONDONTWRITEBYTECODE=1 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PROJECT_ENVIRONMENT=/home/ipol/.uv-envs/dr \
    # Use a throwaway cache under /tmp for both build & runtime
    UV_CACHE_DIR=/tmp/uv-cache

# Create runtime user & dirs (as root)
RUN groupadd -g 1000 ipol \
 && useradd -m -u 1000 -g 1000 -d "$HOME" ipol \
 && mkdir -p /workdir "$bin" /opt/uv/python /home/ipol/.uv-envs "$UV_CACHE_DIR" \
 && rm -rf /home/ipol/.uv-cache  \
 && chown -R ipol:ipol /workdir "$HOME" /opt/uv /home/ipol/.uv-envs "$UV_CACHE_DIR" \
 && chmod -R 755 /opt/uv

# ---- Switch to user BEFORE copying/syncing ----
USER ipol
WORKDIR $bin

# Copy project and make sure /workdir/bin is writable by ipol
COPY --chown=ipol:ipol . .
RUN chown -R ipol:ipol /workdir /workdir/bin

# Prepare cache dir under /tmp (owned by ipol)
RUN rm -rf "$UV_CACHE_DIR" && mkdir -p "$UV_CACHE_DIR"

# Create the environment OUTSIDE the project folder and install deps
RUN uv sync

# QoL
ENV PATH=$bin:$PATH

