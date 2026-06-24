import unittest
import pytest
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from adapter import RustyMinMaxScaler

class TestAdapterMinMaxScaler(unittest.TestCase):
    def test_rusty_scaler(self):
        rng = np.random.default_rng(42)
        shapes = [(1000, 10), (10000, 100), (100000, 100), (1000000, 10), (10000000, 1)]
        
        for shape in shapes:
            x = rng.standard_normal(size=shape, dtype=np.float32) * 10 + 100
            sk = MinMaxScaler()
            sk_out = sk.fit_transform(x)
            rusty = RustyMinMaxScaler()
            rusty_out = rusty.fit_transform(x)
            np.testing.assert_allclose(sk_out,rusty_out, rtol=1e-6, atol=1e-6)
            