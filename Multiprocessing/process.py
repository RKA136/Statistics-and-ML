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