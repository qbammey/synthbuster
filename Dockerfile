# ---- Base image ----
FROM registry.ipol.im/ipol:v2-py3.11

# ---- Install uv binaries ----
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ---- Paths & defaults (no cache vars set yet) ----
ENV HOME=/home/ipol \
    bin=/workdir/bin \
    PYTHONDONTWRITEBYTECODE=1 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    # Put uv-managed Pythons in a user-writable place
    UV_PYTHON_INSTALL_DIR=/home/ipol/.local/share/uv/python \
    # Keep the project environment outside the repo tree
    UV_PROJECT_ENVIRONMENT=/home/ipol/.uv-envs/dr

# ---- Create user and *all* needed dirs as root, then chown ----
RUN groupadd -g 1000 ipol \
 && useradd -m -u 1000 -g 1000 -d "$HOME" ipol \
 && mkdir -p /workdir "$bin" \
           /home/ipol/.uv-envs \
           /home/ipol/.local/share/uv/python \
           /home/ipol/.uv-cache \
           /tmp/uv-build-cache \
 && chown -R ipol:ipol /workdir "$HOME" \
           /home/ipol/.uv-envs \
           /home/ipol/.local/share/uv \
           /home/ipol/.uv-cache \
           /tmp/uv-build-cache \
 && chmod 700 /home/ipol/.uv-cache

# ---- Switch to runtime user before touching the project ----
USER ipol
WORKDIR $bin

# ---- Copy project files owned by ipol ----
COPY --chown=ipol:ipol . .

# ---- Build step: use *throwaway* cache and disable caching ----
ENV UV_NO_CACHE=1 \
    UV_CACHE_DIR=/tmp/uv-build-cache
RUN uv sync

# ---- Runtime settings: enable a clean home cache ----
ENV UV_NO_CACHE=0 \
    UV_CACHE_DIR=/home/ipol/.uv-cache \
    PATH=$bin:$PATH

