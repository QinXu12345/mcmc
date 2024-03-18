from abc import ABC, abstractmethod
import numpy as np
from typing import Literal

class Hamiltonian(ABC):
    def __init__(self, *args):
        self._dim = len(args)
        self._size = args[0] if args else 0
        self._J = 1
        self._B = 0
        self._state = np.random.choice([-1,1],self._size)
        
    @property
    def J(self):
        return self._J
    
    @property
    def B(self):
        return self._B
    
    @property
    def size(self):
        return self._size
    
    @abstractmethod
    def energy(self):
        ...
    
    
class OneDimHamiltonian(Hamiltonian):
    def __init__(
        self,
        first: Literal[-1,1] = -1,
        second: Literal[-1,1] = 1,
        N: int = 60
        ):
        assert N > 0
        super().__init__(N)
        self._state[0] = first
        self._state[-1] = second
        
    @property
    def state(self):
        return self._state
        
    def _dislocate_sum(self):
        a = self._state[:-1]
        b = self._state[1:]
        return np.sum(a*b,axis=0)
    
    def energy(self):
        e = -self._J * self._dislocate_sum() - self._B * np.sum(self._state)
        return e 
    
    def __getitem__(self, key:int):
        return self._state[key]
    
    def __setitem__(self, key:int, value:int):
        self._state[key] = value