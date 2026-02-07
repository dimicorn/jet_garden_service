FROM python:3.11-slim

WORKDIR /core

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY services/core/ .

RUN pip install --no-cache-dir -r requirements.txt && \
    python -m grpc_tools.protoc -I proto --python_out=src \
    --grpc_python_out=src proto/core.proto

CMD ["python", "src/main.py"]