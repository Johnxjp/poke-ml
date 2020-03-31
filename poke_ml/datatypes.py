from typing import Callable, Sequence, Union

import numpy as np
from PIL import Image
from torch import Tensor

NumpyArray = np.ndarray
GenericArray = Union[NumpyArray, Tensor]
ImageObject = Image.Image
ImageTransform = Callable[[ImageObject], ImageObject]
ComposedTorchTransforms = Sequence[Callable[[ImageObject], Tensor]]
