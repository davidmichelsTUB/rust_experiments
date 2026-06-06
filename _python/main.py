import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time

x = np.random.uniform(0, 10, (10_000_000, 5)).astype(np.float32)
scaler = MinMaxScaler()

start = time.perf_counter()
scaler.fit(x)
print(f"took: {time.perf_counter() - start:.4f}s")

print(scaler.data_min_)
print(scaler.data_max_)
