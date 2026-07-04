import sys, os

sys.path.insert(0, os.path.abspath(".."))  # project root

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from adapters.LogisticRegression import LogisticRegression as RustLogisticRegression
import timeit
import pandas as pd