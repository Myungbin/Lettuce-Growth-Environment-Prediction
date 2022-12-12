import numpy as np
import torch
from sdv.tabular import CTGAN

torch.manual_seed(0)
np.random.seed(0)
model = CTGAN(epochs=10)
model.fit(day0)
model.sample(5).describe()
