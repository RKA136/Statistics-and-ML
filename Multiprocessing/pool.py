from multiprocessing import Pool, cpu_count

def square(x):
    return x*x

if __name__ == "__main__":
    data = list(range(10))
    with Pool(processes = cpu_count()) as pool:
        results = pool.map(square, data)
    print("Squared Results:", results)