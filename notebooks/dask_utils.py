from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np
import torch
from nptyping import NDArray
from PIL import Image
from torchtyping import TensorType
from torchvision import transforms


def load_image(path: Union[str, Path]) -> Image.Image:
    with Path(path).open("rb") as infile:
        return Image.open(infile).convert("RGB")


def preprocess_image(
    img: NDArray[(3, Any, Any), np.uint8]
) -> NDArray[(3, 224, 224), np.float64]:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )(img)


def stack_batch(
    batch: Tuple[TensorType[224, 224, 3, torch.float64]]
) -> TensorType[-1, 224, 224, 3, torch.float64]:
    return torch.stack(batch)


def predict(
    batch: TensorType[3, 224, 224, -1, torch.float64], model
) -> NDArray[(Any,), np.float64]:
    with torch.no_grad():
        out = model(batch)
        _, predicted = torch.max(out, 1)
    return predicted.numpy()
