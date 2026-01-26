import numpy as np
from numpy.typing import NDArray
import timeit
from pympler import asizeof
from functools import partial


xy = tuple(i * 1.1 for i in range(10)) + tuple(i for i in range(10)) + tuple(i * -2.1 for i in range(10)) + tuple(-i**2 for i in range(70))
x = tuple(i * 1.1 for i in range(10)) + tuple(i * -2.1 for i in range(10))
y = tuple(i for i in range(10)) + tuple(-i**2 for i in range(70))
assert len(x) + len(y) == len(xy), 'size mismatch'
class Optimizer:

    def __init__(self, var_types):
        self.var_types = var_types
        self.evaluator_args = 'list'

def init_optimizer():
    var_types = []
    for i_var in range(0, 10):
        var_types.append((f'var_{i_var}', np.float64))
    for i_var in range(10, 20):
        var_types.append((f'var_{i_var}', np.int32))
    for i_var in range(20, 30):
        var_types.append((f'var_{i_var}', np.float64))
    for i_var in range(30, 100):
        var_types.append((f'var_{i_var}', np.int32))

    var_types = np.dtype(var_types)
    o = Optimizer(var_types)
    o.dimensions = len(xy)
    o.x_size = len(x)
    o.y_size = len(y)

    return o

def ndarray_to_list(arr):
    return list(arr)
def ndarray_to_list2(arr):
    return arr.tolist()
def list_to_ndarray(lst):
    return np.array(lst)
def list_to_ndarray2(lst):
    return np.asarray(lst)

def list_loop():
    a = list(range(len(xy)))
    for i , v in enumerate(a):
        a[i] += i**2

def ndarray_loop():
    a = np.arange(len(xy))
    for i, v in enumerate(a):
        a[i] += i ** 2

"""
Candidate mixed
"""
class C_mixed():
    def __init__(self, optimizer: Optimizer | None) -> None:
        self.X: NDArray[np.float64] = np.asarray(xy, dtype=optimizer.var_types)
        self.O: NDArray[np.float64] = np.full(2, np.nan)
        self.C: NDArray[np.float64] = np.full(5, np.nan)
        self.f: np.float64 = np.float64(np.nan)

def init_c_mixed(optimizer: Optimizer | None) :
    c = C_mixed(optimizer)
    return c
def calc_c_mixed(c: C_mixed, optimizer: Optimizer):
    X = []
    Y = []
    for k, (var_type, _) in c.X.dtype.fields.items():
        if var_type == np.int32:
            Y.append(c.X[k])
        elif var_type == np.float64:
            X.append(c.X[k])
        else:
            raise NotImplementedError
    X = np.array(X)
    Y = np.array(Y)
    f = np.sum(X ** 2) + np.sum(Y ** 2)
    return f
def manipulate_c_mixed(c: C_mixed, optimizer: Optimizer):
    for k, (var_type, _) in c.X.dtype.fields.items():
        if var_type == np.int32:
            c.X[k] += 1
        elif var_type == np.float64:
            c.X[k] += 0.17
        else:
            raise NotImplementedError
    return c

"""
Candidate float
"""
class C_float():
    def __init__(self, optimizer: Optimizer | None) -> None:
        self.X: NDArray[np.float64] = np.asarray(xy, dtype=np.float64)
        self.O: NDArray[np.float64] = np.full(2, np.nan)
        self.C: NDArray[np.float64] = np.full(5, np.nan)
        self.f: np.float64 = np.float64(np.nan)

def init_c_float(optimizer: Optimizer | None):
    c = C_float(optimizer)
    return c
def calc_c_float(c: C_float, optimizer: Optimizer):
    X = []
    Y = []
    for (_, (var_type, _)), v in zip(o.var_types.fields.items(), c.X):
        if var_type == np.int32:
            Y.append(v)
        elif var_type == np.float64:
            X.append(v)
        else:
            raise NotImplementedError
    X = np.array(X)
    Y = np.array(Y)
    f = np.sum(X ** 2) + np.sum(Y ** 2)
    return f
def manipulate_c_float(c: C_float, optimizer: Optimizer):
    for i, (k, (var_type, _)) in enumerate(o.var_types.fields.items()):
        if var_type == np.int32:
            c.X[i] += 1
        elif var_type == np.float64:
            c.X[i] += 0.17
        else:
            raise NotImplementedError
    return c


"""
Candidate XY
"""
class C_xy():
    def __init__(self, optimizer: Optimizer | None) -> None:
        self.X: NDArray[np.float64] = np.asarray(x, dtype=np.float64)
        self.Y: NDArray[np.int32] = np.asarray(y, dtype=np.int32)
        self.O: NDArray[np.float64] = np.full(2, np.nan)
        self.C: NDArray[np.float64] = np.full(5, np.nan)
        self.f: np.float64 = np.float64(np.nan)

def init_c_xy(optimizer: Optimizer | None):
    c = C_xy(optimizer)
    return c
def calc_c_xy(c: C_xy, optimizer: Optimizer):
    X = np.array(c.X)
    Y = np.array(c.Y)
    f = np.sum(X ** 2) + np.sum(Y ** 2)
    return f
def manipulate_c_xy(c: C_xy, optimizer: Optimizer):
    for i, y in enumerate(c.Y):
        c.Y[i] += 1
    for i, x in enumerate(c.X):
        c.X[i] += 0.17
    return c

"""
C list
"""
class C_list():
    def __init__(self, optimizer: Optimizer | None) -> None:
        self.X: list = list(xy)
        self.O: NDArray[np.float64] = np.full(2, np.nan)
        self.C: NDArray[np.float64] = np.full(5, np.nan)
        self.f: np.float64 = np.float64(np.nan)

def init_c_list(optimizer: Optimizer | None) :
    c = C_list(optimizer)
    return c
def calc_c_list(c: C_list, optimizer: Optimizer):
    X = []
    Y = []
    for var in c.X:
        var_type = type(var)
        if var_type == int:
            Y.append(var)
        elif var_type == float:
            X.append(var)
        else:
            raise NotImplementedError
    X = np.array(X)
    Y = np.array(Y)
    f = np.sum(X ** 2) + np.sum(Y ** 2)
    return f
def manipulate_c_list(c: C_list, optimizer: Optimizer):
    for i, x in enumerate(c.X):
        if type(x) == int:
            c.X[i] += 1
        elif type(x) == float:
            c.X[i] += 0.17
        else:
            raise NotImplementedError
    return c


"""
C dict
"""
class C_dict():
    def __init__(self, optimizer: Optimizer | None) -> None:
        self.X: dict =  {f'var{i}': v for i, v in enumerate(xy)}
        self.O: NDArray[np.float64] = np.full(2, np.nan)
        self.C: NDArray[np.float64] = np.full(5, np.nan)
        self.f: np.float64 = np.float64(np.nan)

def init_c_dict(optimizer: Optimizer | None) :
    c = C_dict(optimizer)
    return c
def calc_c_dict(c: C_dict, optimizer: Optimizer):
    X = []
    Y = []
    for k, var in c.X.items():
        var_type = type(var)
        if var_type == int:
            Y.append(var)
        elif var_type == float:
            X.append(var)
        else:
            raise NotImplementedError
    X = np.array(X)
    Y = np.array(Y)
    f = np.sum(X ** 2) + np.sum(Y ** 2)
    return f
def manipulate_c_dict(c: C_list, optimizer: Optimizer):
    for i, x in c.X.items():
        if type(x) == int:
            c.X[i] += 1
        elif type(x) == float:
            c.X[i] += 0.17
        else:
            raise NotImplementedError
    return c


def benchmark():
    o = init_optimizer()
    N = 1_00_000
    arr = np.random.uniform(size=100)
    lst = list(arr)

    print(f'Elapsed time for {N:_d} repeats')
    print('Design vector: 10 float + 10 int + 10 float + 70 int variables')

    t_lst = timeit.timeit(list_loop, number=N)
    print(f'list loop:        {t_lst:13.3f} s')
    t_arr = timeit.timeit(ndarray_loop, number=N)
    print(f'ndarray loop:     {t_arr:13.3f} s')
    t_arr_to_lst = timeit.timeit(partial(ndarray_to_list, arr), number=N)
    print(f'list(ndarray):    {t_arr_to_lst:13.3f} s')
    t_arr_to_lst2 = timeit.timeit(partial(ndarray_to_list2, arr), number=N)
    print(f'ndarray.tolist:   {t_arr_to_lst2:13.3f} s')
    t_lst_to_arr = timeit.timeit(partial(list_to_ndarray, lst), number=N)
    print(f'np.array(list):   {t_lst_to_arr:13.3f} s')
    t_lst_to_arr2 = timeit.timeit(partial(list_to_ndarray2, lst), number=N)
    print(f'np.asarray(list): {t_lst_to_arr2:13.3f} s')

    print(f'{"approach":>15s}{"t init":>15s}{"t calc":>15s}{"t manipulate":>15s}{"cand. mem.":>15s}')
    for init_c, calc_c, manipulate_c, C in zip([init_c_mixed, init_c_float, init_c_xy, init_c_list, init_c_dict],
                                               [calc_c_mixed, calc_c_float, calc_c_xy, calc_c_list, calc_c_dict],
                                               [manipulate_c_mixed, manipulate_c_float, manipulate_c_xy, manipulate_c_list, manipulate_c_dict],
                                               [C_mixed, C_float, C_xy, C_list, C_dict]):
        print(f'{C.__name__:>15s}', end='')
        p = partial(init_c, o)
        t1 = timeit.timeit(p, number=N)
        print(f'{t1:13.3f} s', end='')
        p = partial(calc_c, C(o), o)
        t2 = timeit.timeit(p, number=N)
        print(f'{t2:13.3f} s', end='')
        p = partial(manipulate_c, C(o), o)
        t3 = timeit.timeit(p, number=N)
        print(f'{t3:13.3f} s', end='')
        print(f'{asizeof.asizeof(C(o)):>9d} bytes', end='')
        print()



class Candidate:

    def __init__(self, optimizer: Optimizer | None) -> None:
        self._X: list = list([])
        if optimizer.evaluator_args == "list":
            self._get_x = self.get_x_as_list
        elif optimizer.evaluator_args == "ndarray":
            self._get_x = self.get_x_as_ndarray
        else:
            raise NotImplementedError

    @property
    def X(self):
        return self._get_x()

    @X.setter
    def X(self, value):
        self._set_x(value)

    def get_x_as_list(self):
        return self._X
    def get_x_as_ndarray(self):
        return np.asarray(self._X, dtype=np.float64)
    def get_x_as_xy(self):
        X = []
        Y = []
        for x in self._X:
            if type(x) in [float, np.float64]:
                X.append(x)
            elif type(x) in [int, np.int32]:
                Y.append(x)
            else:
                raise NotImplementedError
        return np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.int32)

    def _set_x(self, value):
        if type(value) == list:
            self._X = value
        elif type(value) == np.ndarray:
            self._X = value.tolist()




if __name__ == "__main__":

    optimizer = Optimizer()

    variables = {'var1': ('R', -100, 100), # Real (continuous) bounded
                 'var2': ('R', 0, None), # Real (continuous) semi-bounded
                 'var3': ('R', None, None), # Real (continuous) unbounded
                 'var4': ('D', [1.1, 1.2, 1.3, 1.4, 1.5]), # Discrete (float for evaluator, int for optimizer)
                 'var5': ('I', 0, 10), # Integer (bot for optimizer and evaluator)
                 'var6': ('C', ['a', 'b', 'c', 'd', 'e']),  # Category
                 }
    optimizer.variables = variables

    var_types = []
    for i_var in range(0, 3):
        var_types.append((f'var_{i_var}', np.float64))
    for i_var in range(3, 5):
        var_types.append((f'var_{i_var}', np.int32))
    var_types = np.dtype(var_types)
    optimizer = Optimizer(var_types)

    x_lst = [1.1 * i for i in range(3)] + [i**2 for i in range(2)]
    x_arr = np.asarray(x_lst, dtype=np.float64)

    optimizer.evaluator_args = 'list'
    c1 = Candidate(optimizer)
    c1.X = x_lst
    print(f'{c1.X=}')
    print(f'{c1.get_x_as_list()=}')
    print(f'{c1.get_x_as_ndarray()=}')
    print(f'{c1.get_x_as_xy()=}')

    optimizer.evaluator_args = 'ndarray' # default
    c2 = Candidate(optimizer)
    c2.X = x_lst
    print(f'{c2.X=}')
