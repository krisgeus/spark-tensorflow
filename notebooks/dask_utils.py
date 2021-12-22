from pathlib import Path
from typing import Any, List, Tuple, Union

import dask
import numpy as np
import toolz
import torch
from nptyping import NDArray
from PIL import Image
from torchtyping import TensorType
from torchvision import transforms
from torchvision.models.resnet import ResNet


def load_image(path: Union[str, Path]) -> Image.Image:
    with Path(path).open("rb") as infile:
        return Image.open(infile).convert("RGB")


def preprocess_image(
    img: NDArray[(Any, Any, 3), np.uint8]
) -> TensorType[3, 224, 224, torch.float64]:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )(img)


def stack_batch(
    batch: Tuple[TensorType[3, 224, 224, torch.float64]]
) -> TensorType[-1, 3, 224, 224, torch.float64]:
    return torch.stack(batch)


def predict(
    batch: TensorType[-1, 3, 224, 224, torch.float64], model: ResNet
) -> NDArray[(Any, 1000), np.float64]:
    with torch.no_grad():
        out = model(batch)
    return out.numpy()


@dask.delayed
def load_image_wrapper(img_path: Union[Path, str]) -> NDArray[(Any, Any, 3), np.uint8]:
    return load_image(img_path)


@dask.delayed
def preprocess_image_wrapper(
    img: NDArray[(Any, Any, 3), np.uint8]
) -> TensorType[3, 224, 224, torch.float64]:
    return preprocess_image(img)


@dask.delayed
def stack_batch_wrapper(
    batch: Tuple[TensorType[3, 224, 224, torch.float64]]
) -> TensorType[-1, 3, 224, 224, torch.float64]:
    return stack_batch(batch)


@dask.delayed
def predict_wrapper(
    batch: TensorType[-1, 3, 224, 224, torch.float64],
    model: ResNet,
    return_class: bool = False,
) -> Union[NDArray[(Any, 1000), np.float64], NDArray[(Any,), np.int64]]:
    if not return_class:
        return predict(batch, model)
    else:
        return np.argmax(predict(batch, model), axis=1)


def create_batches(
    data: List[TensorType[3, 224, 224, torch.float64]], batch_size: int = 10
) -> TensorType[-1, 3, 224, 224, torch.float64]:
    return [
        stack_batch_wrapper(batch) for batch in toolz.partition_all(batch_size, data)
    ]
