# Why Multiprocessing?
- Python threads are limited by the *Global Interpreter Lock(GIL)* → only one thread executes Python bytecode at a time.
- Multiprocessing bypasses the GIL by creating separate processes, each with its own Python interpreter and memory space.
- Ideal for CPU-bound tasks like numerical computation, data parsing, simulations.

# Core Concepts
Python's `multiprocessing` module provides:
- `Process`: Run a function in a separate process.
- `Pool`: Create a pool of worker processes and distribute tasks.
- `Queue` and `Pipe`: Share data between processes.
- `Manager`: Control shared state safely across processes.

# Simple Example (using `Process`)
```python
import multiprocessing
import os
import time

def worker_task(x):
    print(f"Process {os.getpid} squaring {x}")
    time.sleep(1)
    return x*x

if __name__ == "__main__":
    processes = []
    for i in range(5):
        p = multiprocessing.Process(target=worker_task, args=(i,))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    print("All processes completed.")
```
**Output**
```text
Process <built-in function getpid> squaring 0Process <built-in function getpid> squaring 1

Process <built-in function getpid> squaring 2
Process <built-in function getpid> squaring 3
Process <built-in function getpid> squaring 4
All processes completed.
```

- Launcher **5 Processes**
- Each executes independently.

# Using `Pool` (Easier for Batch Data)

```python
from multiprocessing import Pool, cpu_count

def square(x):
    return x*x

if __name__ == "__main__":
    data = list(range(10))
    with Pool(processes = cpu_count()) as pool:
        results = pool.map(square, data)
    print("Squared Results:", results)
```
**Output**

```text
Squared Results: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

- Uses all availabe CPU cores(`cpu_count()`).
- `map` distributtes work across cores.
- Collects results in order.

# Sharing Data (with `Queue`)

```python
from multiprocessing import Process, Queue

def worker(q,x):
    q.put(x * x)
    
if __name__ == "__main__":
    q = Queue()
    processes = []
    for i in range(5):
        p = Process(target=worker, args=(q,i))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    results = [q.get() for _ in processes]
    print("Squared Results:", results)
```

**Output**

```text
Squared Results: [1, 0, 16, 9, 4]
```
- Queus allows results to be passes back.

# Best Practices
1. Always guard entry with 
```python
if __name__ == "__main__":
```
(especially in Windows)
2. Use `Pool` fir simple paraller loops.
3. Use `Process` + `Queue/Manager` for complex workflows.
4. Don't overscribe cores (`processes > cpu_count()` rarely helps)
5. For large data → avoid picking overhead, use shared memory (`multiprocessing.shared_memory` in Python 3.8+)

# Use `concurrent.futures.ProcessPoolExecutor`
This is a modern, highre-level API than `multiprocessing.Pool`. It's cleaner and integrates well with Python.

```python
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
```
**Output**
```text
PID 19384 processing 0
PID 6612 processing 1
PID 20200 processing 2
PID 18508 processing 3
PID 20408 processing 4
PID 18968 processing 5
PID 12456 processing 6
PID 7408 processing 7
PID 3948 processing 8
PID 16448 processing 9
Results: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```
- Automatically chooses the right number of workers (by default = CPU cores).
- `as_completed` gives you results as soon as they are done (not strictly in order).

# Combine with Numpy (CPU acceleration inside Each Process)

Sometimes a single process doesn't need to just square numbers - It may need to handle big arrays.

```python
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
```
**Output**
```text
Results: [np.float64(1.301141750443172), np.float64(1.3011564435259444), np.float64(1.3012120086330983), np.float64(1.3011731511648024), np.float64(1.301167141282091), np.float64(1.301180566862531), np.float64(1.3012034013376712), np.float64(1.3011600525150862)]
```
Now each process runs a NumPy-heavy computation independently → full CPU usage.