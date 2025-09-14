FROM registry.ipol.im/ipol:v2-py3.11
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create runtime user & dirs (as root)
ENV HOME=/home/ipol
RUN groupadd -g 1000 ipol \
 && useradd -m -u 1000 -g 1000 -d "$HOME" ipol \
 && mkdir -p /workdir /workdir/bin /home/ipol/.uv-envs /opt/uv/python \
 && chown -R ipol:ipol /workdir "$HOME" /home/ipol/.uv-envs \
 && chmod -R 755 /opt/uv   # readable/traversable by all

# Switch to runtime user
USER ipol
ENV bin=/workdir/bin
WORKDIR $bin

# Copy project
COPY --chown=ipol:ipol . .

# uv settings:
# - No cache (avoid permission flakes)
# - Managed interpreters live in /opt/uv/python (world-readable)
# - Project env outside repo
ENV UV_NO_CACHE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PROJECT_ENVIRONMENT=/home/ipol/.uv-envs/dr

# Belt & braces: remove any stray caches
RUN rm -rf /home/ipol/.uv-cache /root/.cache/uv /tmp/uv-cache || true

# Create env (uv will install Python 3.13 to /opt/uv/python per >=3.12 and .python-version=3.13)
RUN uv sync

# QoL
ENV PYTHONDONTWRITEBYTECODE=1 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    PATH=$bin:$PATH

