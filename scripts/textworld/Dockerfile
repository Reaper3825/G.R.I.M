FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.title="GRIM TextWorld environment"
LABEL org.opencontainers.image.version="1.7.0"

ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# Native prerequisites for Python dependency builds. No GRIM build targets.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/grim-textworld/requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/grim-textworld/requirements.txt \
    && python -m pip check \
    && python -m pip freeze > /opt/grim-textworld/installed-packages.txt

COPY smoke_test.py /opt/grim-textworld/smoke_test.py
RUN useradd --create-home --uid 10001 textworld
USER textworld
WORKDIR /home/textworld

# Standalone dependency verification only; no service or inference integration.
CMD ["python", "/opt/grim-textworld/smoke_test.py"]
