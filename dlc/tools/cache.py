import abc
import multiprocessing.managers
from typing import Any
from typing import Callable
from typing import Optional
from typing import Union

try:
    from multiprocessing import shared_memory
except Exception:
    # Use backport
    import shared_memory

import numpy as np


CacheDictType = Union[multiprocessing.managers.DictProxy, dict]


class BaseArrayCache:
    def __init__(self, cache_dict: Optional[CacheDictType] = None):
        if cache_dict is not None:
            self._cache = cache_dict
        else:
            self._cache = dict()
        self._initialize()

    @property
    def hits(self):
        return self._cache["__hits"]

    @property
    def misses(self):
        return self._cache["__misses"]

    @property
    def updates(self):
        return self._cache["__updates"]

    @property
    def size(self):
        return self._cache["__size"]

    @property
    def keys(self):
        return self._cache.keys()

    @abc.abstractmethod
    def read(self, key: str) -> Optional[np.ndarray]:
        pass

    @abc.abstractclassmethod
    def write(self, key: str, value: np.ndarray) -> None:
        pass

    def get_or_create(
        self,
        key: str,
        callable: Callable[[], np.ndarray],
    ) -> np.ndarray:
        value = self.read(key)
        if value is None:
            value = callable()
            self.write(key, value)
        return value


class ArrayCache(BaseArrayCache):
    def _initialize(self):
        if "__initialized" not in self._cache:
            self._cache["__initialized"] = True
            self._cache["__hits"] = 0
            self._cache["__misses"] = 0
            self._cache["__updates"] = 0
            self._cache["__size"] = 0
            self._cache["__n_entries"] = 0

    def read(self, key: str) -> Optional[np.ndarray]:
        if key in self._cache:
            self._cache["__hits"] += 1
        else:
            self._cache["__misses"] += 1

        return self._cache.get(key, None)

    def write(self, key: str, value: np.ndarray) -> None:
        if key not in self._cache:
            self._cache["__updates"] += 1
        else:
            self._cache["__size"] -= self._cache[key].nbytes
        self._cache["__size"] += value.nbytes
        self._cache[key] = value

    def clear(self):
        self._cache.clear()
        self._initialize()

    def __len__(self):
        return len(self._cache) - 6


class SharedArrayCache(BaseArrayCache):
    # Assumes underlying dictionary is a DictProxy instance.
    # See: https://docs.python.org/3/library/multiprocessing.shared_memory.html

    def _initialize(self):
        if "__initialized" not in self._cache:
            self._cache["__initialized"] = True
            self._cache["__hits"] = 0
            self._cache["__misses"] = 0
            self._cache["__updates"] = 0
            self._cache["__size"] = 0
            self._cache["__n_entries"] = 0
            self._cache["__names"] = dict()
            self._cache["__shapes"] = dict()
            self._cache["__dtypes"] = dict()

    def read(self, key: str) -> Optional[np.ndarray]:
        if key in self._cache:
            self._cache["__hits"] += 1
            shm = shared_memory.SharedMemory(name=self._cache["__names"][key])
            shape = self._cache["__shapes"][key]
            dtype = self._cache["__dtypes"][key]
            value = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            shm.close()
            return value
        else:
            self._cache["__misses"] += 1
        return None

    def write(self, key: str, value: np.ndarray) -> None:
        if key not in self._cache:
            self._cache["__updates"] += 1
        else:
            shm = shared_memory.SharedMemory(name=self._cache["__names"][key])
            self._cache["__size"] -= shm.size
            shm.close()
            shm.unlink()
        shm = shared_memory.SharedMemory(create=True, size=value.nbytes)
        shm_arr = np.ndarray(value.shape, dtype=value.dtype, buffer=shm.buf)
        shm_arr[:] = value[:]
        self._cache["__size"] += value.nbytes
        self._cache["__names"][key] = shm.name
        self._cache["__shapes"][key] = value.shape
        self._cache["__dtypes"][key] = value.dtype
        shm.close()

    def clear(self):
        for name in self._cache["__names"]:
            shm = shared_memory.SharedMemory(name=name)
            shm.close()
            shm.unlink()
        self._cache.clear()
        self._initialize()

    def __len__(self):
        return len(self._cache) - 9


class DummyCache(BaseArrayCache):
    def _initialize(self):
        pass

    def read(self, key: str) -> Optional[np.ndarray]:
        raise NotImplementedError()

    def write(self, key: str, value: np.ndarray) -> None:
        raise NotImplementedError()

    def get_or_create(
        self,
        key: str,
        callable: Callable[[], np.ndarray],
    ) -> np.ndarray:
        return callable()
