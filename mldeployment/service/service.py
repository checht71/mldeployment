"""This module defines a BentoML service that uses a Keras model to classify
digits.
"""

import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

BENTO_MODEL_TAG = "torch_model_86:26bscybtekyarrdv"


classifier_runner = bentoml.pytorch.get(BENTO_MODEL_TAG).to_runner()

torch_service = bentoml.Service("torch_service", runners=[classifier_runner])

@torch_service.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_data: np.ndarray) -> np.ndarray:
    return classifier_runner.predict.run(input_data)
