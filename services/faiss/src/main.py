from concurrent.futures import ThreadPoolExecutor
import os
import logging
import grpc
from faiss_pb2_grpc import add_FaissServiceServicer_to_server
from faiss_service import FaissService


logger = logging.getLogger(__name__)


def serve() -> None:
    host = os.getenv("FAISS_HOST", "0.0.0.0")
    port = int(os.getenv("FAISS_PORT", "50051"))
    num_workers = int(os.getenv("FAISS_NUM_WORKERS"))
    db_path = os.getenv("FAISS_DB_PATH", "0.0.0.0")
    map_size = int(os.getenv("FAISS_MAP_SIZE", 1024 * 1024 * 100))  # 100 MB

    server = grpc.server(ThreadPoolExecutor(max_workers=num_workers))
    add_FaissServiceServicer_to_server(FaissService(db_path, map_size), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logger.info(f"gRPC server started on {host}:{port}")
    server.wait_for_termination()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    serve()


if __name__ == "__main__":
    main()
