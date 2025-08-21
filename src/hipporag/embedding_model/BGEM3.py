from copy import deepcopy
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError as e:
    BGEM3FlagModel = None

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig

logger = get_logger(__name__)


class BGEM3EmbeddingModel(BaseEmbeddingModel):
    """Embedding wrapper for BAAI/bge-m3 via FlagEmbedding.

    Produces dense embeddings only (default behavior for HippoRAG).
    """

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")

        self._init_embedding_config()

        if BGEM3FlagModel is None:
            raise ImportError(
                "FlagEmbedding is required for BGEM3EmbeddingModel. Please install via `pip install FlagEmbedding`."
            )

        # Choose device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        use_fp16 = (self.global_config.embedding_model_dtype in ("float16", "auto")) and device == "cuda"

        logger.debug(
            f"Initializing {self.__class__.__name__} with device={device}, use_fp16={use_fp16}, model={self.embedding_model_name}"
        )
        self.embedding_model = BGEM3FlagModel(
            self.embedding_model_name,
            use_fp16=use_fp16,
            device=device,
        )

        # bge-m3 dense embedding size is 1024
        self.embedding_dim = 1024

    def _init_embedding_config(self) -> None:
        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            "model_init_params": {
                "model_name_or_path": self.embedding_model_name,
            },
            "encode_params": {
                "batch_size": self.global_config.embedding_batch_size,
                "max_length": self.global_config.embedding_max_seq_len,
                "instruction": "",
            },
        }
        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    def batch_encode(self, texts: List[str], **kwargs):
        if isinstance(texts, str):
            texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs:
            params.update(kwargs)

        batch_size = int(params.pop("batch_size", 16))
        max_length = int(params.pop("max_length", 2048))

        # Instruction is ignored for bge-m3 dense encoding by default.

        logger.debug(f"Calling {self.__class__.__name__} with batch_size={batch_size}, max_length={max_length}")

        def _encode_batch(batch_texts: List[str]):
            # Only pass supported kwargs to avoid leaking flags into tokenizer
            out = self.embedding_model.encode(
                batch_texts,
                max_length=max_length,
                return_dense=True,
                normalize_embeddings=False,  # we will normalize manually if needed
            )
            # FlagEmbedding returns a dict with key 'dense_vecs'
            dense = out["dense_vecs"]
            if isinstance(dense, torch.Tensor):
                dense = dense.cpu().numpy()
            return dense

        # Use the model's internal multiprocessing and batching for stability
        res = self.embedding_model.encode(
            texts,
            max_length=max_length,
            return_dense=True,
        )
        results = res["dense_vecs"] if isinstance(res, dict) else res
        if isinstance(results, torch.Tensor):
            results = results.cpu().numpy()

        if self.embedding_config.norm:
            # L2 normalize rows
            results = (results.T / (np.linalg.norm(results, axis=1, keepdims=False) + 1e-12)).T

        return results


