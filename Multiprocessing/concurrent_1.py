from concurrent.futures import ProcessPoolExecutor, as_completed
import os, time

def heavy_task(x):
    print(f"PID {os.getpid()} processing {x}")
    time.sleep(1) # Simulate a time-consuming task
    return x * x

if __name__ == "__main__":
    data = range(10)
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(heavy_task, x) for x in data]
        for f in as_completed(futures):
            results.append(f.result())
    print("Results:", results)