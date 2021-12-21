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

# Using dask for inference of deep neural networks

See [this Dask tutorial](https://examples.dask.org/machine-learning/torch-prediction.html) for the inspiration of the implementation below.

```python
from typing import Union, Any, Tuple, List
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from nptyping import NDArray
from torchtyping import TensorType
import dask
import torch 
from torchvision import models

%load_ext autoreload
%reload_ext autoreload
%autoreload 2
from dask_utils import (
    load_image_wrapper, 
    predict_wrapper, 
    stack_batch_wrapper, 
    preprocess_image_wrapper, 
    create_batches
)
```

### Load images

```python
images_dir = Path("../data/images/vanalleswat")
images_paths = [img for img in images_dir.glob("*") if img.suffix in ['.jpg', '.png']]
```

Lazy-load the images by calling the delayed `load_image` function (in `dask_utils.py`)

```python
delayed_images = [load_image_wrapper(img) for img in images_paths]
```

```python
delayed_images_preprocessed = [preprocess_image_wrapper(img) for img in delayed_images]
```

Since we've transformed & cropped each image, we should get back an image of shape `n_channels x height x width`

```python
delayed_images_preprocessed[0].compute().shape
```

It makes more sense to batch the images in groups of e.g. 10 and let the model predict the class for each image in a batch.

```python
delayed_images_batched = create_batches(delayed_images_preprocessed, batch_size=10)
```

```python
array_images = [
    image.compute()
    for image_idx, image 
    in enumerate(delayed_images) 
    if image_idx <= 15
]
```

```python
fig, ax = plt.subplots(4, 4, figsize=(12,12))
ax = ax.ravel()

for ax_idx, ax in enumerate(ax):
    ax.imshow(array_images[ax_idx])

plt.tight_layout()
plt.show()
```

## Set up the model


To avoid having to load the model on each worker, we turn the pre-trained resnet model into a delayed object, which allows it to be passed around to each worker as necessary

```python
delayed_model = dask.delayed(models.resnet50(pretrained=True).cpu().eval())
```

```python
delayed_model
```

```python
delayed_batch_predictions = [
    predict_wrapper(batch, delayed_model, return_class=False) 
    for batch 
    in delayed_images_batched
]
```

```python
delayed_predictions = dask.delayed(lambda x: np.concatenate(x, axis=0))(delayed_batch_predictions)
```

```python
# Fetch the results in main memory
predictions = delayed_predictions.compute()
```

I checked and this mapping from TF seems to work fine for pytorch resnet models

```python
class_predictions = tf.keras.applications.resnet50.decode_predictions(
        predictions, top=5
)
```

```python
fig, ax = plt.subplots(4, 4, figsize=(12,12))
ax = ax.ravel()

for img_idx in range(16):

    ax[img_idx].imshow(delayed_images[img_idx].compute())
    ax[img_idx].set_title(class_predictions[img_idx][0][1])

plt.tight_layout()
plt.show()
```

Clearly, this requires some fine-tuning :-).
