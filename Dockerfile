FROM registry.ipol.im/ipol:v2-py3.11

# uv binaries
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# --- Paths (keep cache paths out of HOME) ---
ENV HOME=/home/ipol \
    bin=/workdir/bin \
    PYTHONDONTWRITEBYTECODE=1 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    UV_PYTHON_INSTALL_DIR=/home/ipol/.local/share/uv/python \
    UV_PROJECT_ENVIRONMENT=/home/ipol/.uv-envs/dr \
    # Send *all* caches to /tmp and disable caching
    XDG_CACHE_HOME=/tmp/xdg-cache \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_NO_CACHE=1

# --- Create user & dirs as root, hand them over to ipol ---
RUN groupadd -g 1000 ipol \
 && useradd -m -u 1000 -g 1000 -d "$HOME" ipol \
 && mkdir -p /workdir "$bin" \
            /home/ipol/.uv-envs \
            /home/ipol/.local/share/uv/python \
            /tmp/uv-cache /tmp/xdg-cache \
 && rm -rf /home/ipol/.uv-cache /home/ipol/.cache /root/.cache/uv /root/.uv-cache || true \
 && chown -R ipol:ipol /workdir "$HOME" /tmp/uv-cache /tmp/xdg-cache \
 && chmod -R 755 /home/ipol/.local

# --- Switch to runtime user before touching the project ---
USER ipol
WORKDIR $bin

# Copy project files (owned by ipol)
COPY --chown=ipol:ipol . .

# Create environment (uv will install Python per pyproject/.python-version)
RUN uv sync

# PATH/QoL
ENV PATH=$bin:$PATH

