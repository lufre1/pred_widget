import os
from typing import Callable, Optional, Tuple
import torch_em
import torch
import numpy as np
from torch_em.util.prediction import predict_with_halo


def get_model(model_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.path.exists(model_path):
        if ".pt" in model_path:
            model_path = os.path.dirname(model_path)
        model = torch_em.util.load_model(checkpoint=model_path, device=device)
        print("Model loaded from checkpoint:", model_path)
        return model
    else:
        print(f"Model checkpoint not found at {model_path}.")
        return None


def run_prediction(
    input: np.ndarray,
    model: torch.nn.Module,
    block_shape: Tuple[int, int, int],
    halo: Tuple[int, int, int],
    device: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    input = torch_em.transform.raw.standardize(input)
    return predict_with_halo(
        input_=input,
        model=model,
        gpu_ids=[device],
        block_shape=block_shape,
        halo=halo,
    )
