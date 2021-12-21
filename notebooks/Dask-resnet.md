---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.0
  kernelspec:
    display_name: 'Python 3.8.6 64-bit (''.venv'': poetry)'
    language: python
    name: python3
---

# Deploying a dask cluster for inference of deep neural networks

```python
from typing import Union, Any, Tuple, List
from pathlib import Path

import numpy as np
from nptyping import NDArray
from torchtyping import TensorType
import dask
import dask.array as da
import torch 
import matplotlib.pyplot as plt

import dask_image.imread
import dask_image.ndfilters
import dask_image.ndmeasure

import toolz


%load_ext autoreload
%reload_ext autoreload
%autoreload 2
from dask_utils import load_image, preprocess_image, stack_batch
```

```python
img_dir = Path("../data/images/vanalleswat")
imgs = [img for img in img_dir.glob("*") if img.suffix in ['.jpg', '.png']]
```

Lazy-load the images using dask-image

```python
inference_images = dask_image.imread.imread(img_dir / "*")
```

```python
inference_images
```

```python
fig, ax = plt.subplots(5, 5, figsize=(12, 12))
ax = ax.ravel()

for ax_idx, ax_sub in enumerate(ax):
    ax_sub.imshow(inference_images[ax_idx])
plt.show()
```

We can wrap the loading & preprocessing functions in delayed decorators provided by dask to make them lazy.

```python
@dask.delayed
def load_image_wrapper(img_path: Union[Path, str]) -> NDArray[(1, Any, Any, 3), np.uint8]:
    return load_image(img_path)


@dask.delayed
def preprocess_image_wrapper(img: NDArray[(Any, Any, 3), np.uint8]) -> TensorType[3, 224, 224, torch.float64]:
    return preprocess_image(img)


@dask.delayed
def stack_batch_wrapper(batch: Tuple[TensorType[224, 224, 3, torch.float64]]) -> TensorType[-1, 224, 224, 3, torch.float64]:
    return stack_batch(batch)


def create_batches(data: List[TensorType[224, 224, 3, torch.float64]], batch_size: int = 10) -> TensorType[-1, 224, 224, 3, torch.float64]:
    return [stack_batch_wrapper(batch) for batch in toolz.partition_all(batch_size, data)]
```

```python
imgs_delayed = [load_image_wrapper(img) for img in imgs]
```

```python
imgs_preprocessed = [preprocess_image_wrapper(img) for img in imgs_delayed]
```

```python
imgs_preprocessed[0].compute().shape
```

```python

```

```python
create_batches(imgs_preprocessed, batch_size=10)
```

```python
from dask import Delayed
```

```python

```
