import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
from .adapter import RustyMinMaxScaler

x = np.random.uniform(4, 10, (10_000_000, 5)).astype(np.float32)
scaler = MinMaxScaler()
start = time.perf_counter()
scaler.fit(x)
x = scaler.transform(x)
print(f"took: {time.perf_counter() - start:.4f}s")


scaler = RustyMinMaxScaler()
start = time.perf_counter()
scaler.fit(x)
x = scaler.transform(x)
print(f"took: {time.perf_counter() - start:.4f}s")
