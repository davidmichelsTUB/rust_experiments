from adapter import RustyMinMaxScaler
import numpy as np
import time
import adapter
print("HAVE_RUST:", adapter.HAVE_RUST)
rows = 10_000_000
cols = 50
rng = np.random.default_rng(420)
data = rng.standard_normal(size=(rows,cols))
print(data.shape)
scaler = RustyMinMaxScaler()
time.sleep(1)
print("Starting Fit")
scaler.fit(data)
scaler.fit(data)
scaler.fit(data)
scaler.fit(data)
scaler.fit(data)
print("Finished Fitting")
time.sleep(3)
print("Starting Transform")
scaler.transform(data)
scaler.transform(data)
scaler.transform(data)
scaler.transform(data)
scaler.transform(data)
print("Finished Transform")