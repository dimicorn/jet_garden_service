from concurrent.futures import ThreadPoolExecutor
import os
import logging
import grpc
from faiss_pb2_grpc import add_FaissServiceServicer_to_server
from faiss_service import FaissService


logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    host = os.getenv("FAISS_HOST", "0.0.0.0")
    port = int(os.getenv("FAISS_PORT", "50051"))
    num_workers = int(os.getenv("FAISS_NUM_WORKERS"))
    server = grpc.server(ThreadPoolExecutor(max_workers=num_workers))
    add_FaissServiceServicer_to_server(FaissService(), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logger.info(f"gRPC server started on {host}:{port}")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
