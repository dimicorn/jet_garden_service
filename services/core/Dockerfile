FROM python:3.11-slim

WORKDIR /core

COPY . .

RUN pip install --no-cache-dir -r requirements.txt && \
    python -m grpc_tools.protoc -I proto --python_out=src \
    --grpc_python_out=src proto/core.proto

EXPOSE 50051

ENV CONFIG_PATH config.yaml

CMD ["python", "src/main.py"]