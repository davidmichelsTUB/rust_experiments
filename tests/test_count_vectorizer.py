import unittest
import pytest
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from adapter import RustyCountVectorizer

class TestAdapterMinMaxScaler(unittest.TestCase):
    def test_rusty_scaler(self):
        
        sk = MinMaxScaler()
        sk_out = sk.fit_transform(x)
        rusty = RustyMinMaxScaler()
        rusty_out = rusty.fit_transform(x)
        np.testing.assert_allclose(sk_out,rusty_out, rtol=1e-6, atol=1e-6)
            