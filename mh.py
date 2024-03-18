from hamiltonian import OneDimHamiltonian,Hamiltonian
import numpy as np
from numba import njit

class Ising():
    def __init__(self):
        self._hmt = OneDimHamiltonian()
        self._states = []
        self._total_run = 0
        self._period = 100
    
    def mh_ratio(self):
        spin = np.random.randint(1, self._hmt.size-1)
        ratio = -2 * self._hmt[spin] * (self._hmt.J * (self._hmt[spin-1] + self._hmt[spin + 1]) + self._hmt.B)
        return ratio,spin
    
    def update(self,beta:float):
        ratio,spin = self.mh_ratio()
        criterion = min(1,np.exp(beta * ratio))
        uniform = np.random.uniform(size=1)[0]
        if criterion > uniform:
            self._hmt[spin] = -self._hmt[spin]
        
    def mh(self,epoch: int = 1000,beta:float = 0.2):
        self._total_run = epoch
        self._states.append(self._hmt.state.copy())
        for i in range(epoch):
            self.update(beta)
            if i > 20:
                self._states.append(self._hmt.state.copy())
            
    def tolist(self):
        return self._states
    
    
        
        
        