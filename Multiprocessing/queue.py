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