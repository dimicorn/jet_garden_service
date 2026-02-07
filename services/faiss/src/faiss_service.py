from logging import getLogger
import grpc
import faiss
import numpy as np
from faiss_pb2 import (
    RetrieveTopKRequest,
    RetrieveTopKResponse,
    LoadEmbeddingRequest,
    LoadEmbeddingResponse,
)
from faiss_pb2_grpc import FaissServiceServicer
from db import LMDB


logger = getLogger(__name__)


class FaissService(FaissServiceServicer):
    def __init__(self, db_path: str, map_size: int):
        self.embeddings_db = LMDB(db_path, map_size)
        self.index = None

    @staticmethod
    def _get_index(embeddings: np.ndarray, image_ids: np.ndarray, dim: int):
        faiss.normalize_L2(embeddings)
        base_index = faiss.IndexFlatIP(dim)
        index = faiss.IndexIDMap2(base_index)
        index.add_with_ids(embeddings, image_ids)
        logger.info("created a faiss index")
        return index

    def LoadEmbedding(
        self, request_iterator: LoadEmbeddingRequest, context
    ) -> LoadEmbeddingResponse:
        image_ids = []
        embs = []
        dim = None

        for request in request_iterator:
            if dim is None:
                dim = request.dim
            image_ids.append(request.image_id)
            embs.append(np.asarray(request.embedding, dtype=np.float32))
        embeddings = np.ascontiguousarray(np.vstack(embs), dtype=np.float32)
        image_ids = np.asarray(image_ids, dtype=np.int64)

        self.index = FaissService._get_index(embeddings, image_ids, dim)
        self.embeddings_db.create(
            {
                image_id: embeddings[i].tobytes()
                for i, image_id in enumerate(image_ids.tolist())
            }
        )
        return LoadEmbeddingResponse()

    def RetrieveTopK(
        self, request: RetrieveTopKRequest, context
    ) -> RetrieveTopKResponse:
        if not request.embedding:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "image is empty")
        k = int(request.top_k or 0)
        if k <= 0:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "top_k must be > 0")
        embedding = np.asarray(request.embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(embedding)
        if self.index is not None:
            _, inds = self.index.search(embedding, k)
        else:
            raise TypeError()
        logger.info("retrieving top k")
        return RetrieveTopKResponse(model_version="test", indexes=inds[0])
