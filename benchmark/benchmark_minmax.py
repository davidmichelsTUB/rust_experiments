from adapter import RustyMinMaxScaler
from sklearn.preprocessing import MinMaxScaler
import time
import numpy as np
from itertools import product
from tqdm import tqdm
SEED = 67
N_WARMUP_RUNS = 5
N_REPEATS = 5
n_cols_list = [10,50,100]
n_rows_list = [10_000,100_000,1_000_000,10_000_000]


def _get_data(n_rows: int,n_cols:int,seed = SEED):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=(n_rows, n_cols), dtype=np.float32)


def warmup():
    for _ in range(N_WARMUP_RUNS):
        SKScaler = MinMaxScaler()
        _time_fit_transform(SKScaler,_get_data(n_rows_list[2], n_cols_list[1]))
        RustScaler = RustyMinMaxScaler()
        _time_fit_transform(RustScaler,_get_data(n_rows_list[2], n_cols_list[1]))


        
def _time_fit_transform(scaler, X: np.ndarray) -> tuple[float, float]:
    
    t0 = time.perf_counter()
    scaler.fit(X)
    t1 = time.perf_counter()
    fit_time = t1 - t0

    t0 = time.perf_counter()
    scaler.transform(X)
    t1 = time.perf_counter()
    transform_time = t1 - t0
    return fit_time, transform_time

def main():
    warmup()
    with open("benchmark.csv","w+") as f:
            f.write("n_rows,n_cols,baseline_fit,baseline_transform,rust_fit,rust_transform")
    for n_cols,n_rows in product(n_cols_list,n_rows_list):
        print(f"Processing {n_cols,n_rows}")
        data = _get_data(n_rows,n_cols)
        results = []
        for _ in range(N_REPEATS):
            SKScaler = MinMaxScaler()
            fit_sk, transform_sk = _time_fit_transform(SKScaler,data)
            RustScaler = RustyMinMaxScaler()
            fit_rust, transform_rust = _time_fit_transform(RustScaler,data)
            results.append([fit_sk, transform_sk, fit_rust, transform_rust])
        content = np.mean(results, axis=0)
        with open("benchmark.csv","a") as f:
            f.write(f"\n{n_rows},{n_cols},")
            f.write(",".join(f"{v*1000:.4f}" for v in content))
            

if __name__ == "__main__":
    main()
