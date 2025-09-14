FROM registry.ipol.im/ipol:v2-py3.11
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV HOME=/home/ipol \
    bin=/workdir/bin \
    UV_CACHE_DIR=/home/ipol/.uv-cache \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python
# optional last line: world-readable interpreters

RUN groupadd -g 1000 ipol \
 && useradd -m -u 1000 -g 1000 -d "$HOME" ipol \
 && mkdir -p /workdir /opt/uv/python "$UV_CACHE_DIR" \
 && chown -R ipol:ipol /workdir "$HOME" /opt/uv "$UV_CACHE_DIR" \
 && chmod -R 755 /opt/uv

USER ipol
WORKDIR $bin
COPY --chown=ipol:ipol . .

# Create venv and cache owned by ipol
RUN uv sync

ENV PYTHONDONTWRITEBYTECODE=1 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    PATH=$bin:$PATH

