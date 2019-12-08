import multiprocessing
from multiprocessing import Pool


def f(x):
    return x * x


if __name__ == '__main__':
    cpu_count = multiprocessing.cpu_count()
    with Pool(cpu_count) as p:
        print(p.map(f, [1, 2, 3]))
