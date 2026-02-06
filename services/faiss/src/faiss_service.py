import grpc
from faiss_pb2 import RetrieveTopKRequest, RetrieveTopKResponse
from faiss_pb2_grpc import FaissServiceServicer


class FaissService(FaissServiceServicer):
    def RetrieveTopK(self, request: RetrieveTopKRequest, context):
        if not request.image:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "image is empty")
        k = int(request.top_k or 0)
        if k <= 0:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "top_k must be > 0")

        # TODO: decode image, run torch model -> embedding, call faiss search, return ids/scores
        # matches = [
        #     faiss_pb2.Match(image_id="dummy_1", score=0.99),
        #     faiss_pb2.Match(image_id="dummy_2", score=0.97),
        # ][:k]

        return RetrieveTopKResponse(model_version="test", index_name="-1")
