from pathlib import Path
from typing import Any, Union

import numpy as np
from nptyping import NDArray
from PIL import Image
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
