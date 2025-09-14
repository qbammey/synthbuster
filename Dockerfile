# ---- Base image with Python 3.11 ----
FROM registry.ipol.im/ipol:v2-py3.11

# ---- Install uv ----
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ---- Environment setup ----
ENV HOME=/home/ipol \
    bin=/workdir/bin \
    UV_CACHE_DIR=/home/ipol/.uv-cache \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PROJECT_ENVIRONMENT=/home/ipol/.uv-envs/dr \
    PYTHONDONTWRITEBYTECODE=1 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Create runtime user and directories, set ownership/permissions
RUN groupadd -g 1000 ipol \
 && useradd -m -u 1000 -g 1000 -d "$HOME" ipol \
 && mkdir -p /workdir "$bin" /opt/uv/python "$UV_CACHE_DIR" /home/ipol/.uv-envs \
 && chown -R ipol:ipol /workdir "$HOME" /opt/uv "$UV_CACHE_DIR" /home/ipol/.uv-envs \
 && chmod -R 755 /opt/uv

# ---- Switch to runtime user before copying code / syncing ----
USER ipol
WORKDIR $bin

# Copy project files; ensure directory itself is owned by ipol
COPY --chown=ipol:ipol . .
RUN chown -R ipol:ipol /workdir /workdir/bin

# Ensure a clean, writable uv cache for the build step
RUN rm -rf "$UV_CACHE_DIR" && mkdir -p "$UV_CACHE_DIR"

# Create the environment OUTSIDE the project folder and install deps
RUN uv sync

# QoL: put project bin on PATH
ENV PATH=$bin:$PATH

