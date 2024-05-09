"""This module defines a BentoML service that uses a Keras model to classify
digits.
"""

import numpy as np
import bentoml
from torch import Tensor

BENTO_MODEL_TAG = "torch_model:k45ivbansohbrrdv"


classifier_runner = bentoml.pytorch.get(BENTO_MODEL_TAG).to_runner()

mnist_service = bentoml.Service("pothole_classifier_v0_5", runners=[classifier_runner])

@mnist_service.api(input=Tensor, output=Tensor)
def classify(input_data: Tensor) -> Tensor:
    return classifier_runner.predict.run(input_data)
