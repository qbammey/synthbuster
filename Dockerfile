FROM registry.ipol.im/ipol:v2-py3.11

# Install uv binaries
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# --- Paths and defaults ---
ENV HOME=/home/ipol \
    bin=/workdir/bin \
    PYTHONDONTWRITEBYTECODE=1 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    # Put uv-managed Pythons somewhere ipol can write
    UV_PYTHON_INSTALL_DIR=/home/ipol/.local/share/uv/python \
    # Keep the project env outside the repo tree
    UV_PROJECT_ENVIRONMENT=/home/ipol/.uv-envs/dr \
    # Build without caches to dodge stale permission artifacts
    UV_NO_CACHE=1 \
    # Use a throwaway cache dir during build
    UV_CACHE_DIR=/tmp/uv-build-cache

# --- Create user and writable dirs as root ---
RUN groupadd -g 1000 ipol \
 && useradd -m -u 1000 -g 1000 -d "$HOME" ipol \
 && mkdir -p /workdir "$bin" "$UV_PYTHON_INSTALL_DIR" /home/ipol/.uv-envs "$UV_CACHE_DIR" \
 && chown -R ipol:ipol /workdir "$HOME" "$UV_PYTHON_INSTALL_DIR" /home/ipol/.uv-envs "$UV_CACHE_DIR"

# --- Switch to runtime user before touching project ---
USER ipol
WORKDIR $bin

# Copy project files owned by ipol
COPY --chown=ipol:ipol . .

# Ensure build cache dir exists and is clean
RUN rm -rf "$UV_CACHE_DIR" && mkdir -p "$UV_CACHE_DIR"

# Install: uv will (a) install CPython per pyproject/.python-version into $UV_PYTHON_INSTALL_DIR,
#          (b) create the env at $UV_PROJECT_ENVIRONMENT, all writable by ipol.
RUN uv sync

# Optional: for runtime you can re-enable caching and use a home cache dir
ENV UV_NO_CACHE=0 \
    UV_CACHE_DIR=/home/ipol/.uv-cache \
    PATH=$bin:$PATH

