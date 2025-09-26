import numpy as np
from concurrent.futures import ProcessPoolExecutor

def process_array(seed):
    rng = np.random.default_rng(seed)
    arr = rng.random(10_000_000)
    return np.mean(np.sin(arr) + np.cos(arr))

if __name__ == "__main__":
    seeds = range(8)  # Different seeds for different processes
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_array, seeds))
    print("Results:", results)