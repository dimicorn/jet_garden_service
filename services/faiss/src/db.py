from logging import getLogger
import lmdb
import numpy as np


logger = getLogger(__name__)


class LMDB:
    def __init__(self, db_path: str, map_size: int):
        self.db_path = db_path
        self.map_size = map_size

    def create(self, data: dict[int, np.ndarray]):
        env = lmdb.open(
            self.db_path,
            map_size=self.map_size,
        )

        with env.begin(write=True) as txn:
            for key, embedding in data.items():
                txn.put(key.encode("utf-8"), embedding.tobytes())

        env.close()
        logger.info("LMDB created.")
