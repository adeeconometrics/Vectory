from __future__ import annotations 
from typing import Union, Optional

import numpy as np
import math as m

class FSym: 
    def __init__(self, 
                val:Union[float, np.ndarray], 
                dot:Optional[Union[float, np.nadaaray]] = None) -> None:
        self.val, self.dot = val, dot  

    def __add__(self, other:FSym) -> FSym: 
        df  = FSym(self.val + other.val, self.dot + other.dot)
        return df 

    def __sub__(self, other:FSym) -> FSym: 
        df  = FSym(self.val - other.val, self.dot - other.dot)
        return df 

    def __mul__(self, other:FSym) -> FSym: 
        df = FSym(self.val * other.val,
                    self.val*other.dot + other.val*self.dot)
        return df

    def __truediv__(self, other:FSym) -> FSym: 
        df = FSym(self.val / other.val,
                (self.dot*other.val - self.val*other.dot)/ pow(other.val, 2))
        return df

def sin(x:FSym) -> FSym:
    df = FSym(np.sin(x.val), 
            np.cos(x.val)*x.dot)
    return df

def cos(x:FSym) -> FSym:
    df = FSym(np.cos(x.val), 
            -np.sin(x.val)*x.dot)
    return df

def tan(x:FSym) -> FSym:
    df = FSym(np.tan(x.val), 
            np.sec(x.val)**2 * x.dot)
    return df

def cot(x:FSym) -> FSym: 
    df = FSym(np.tan(x.val), 
            -np.csc(x.val)**2 * x.dot)
    return df

def sec(x:FSym) -> FSym: 
    df = FSym(np.sec(x.val), 
            np.tan(x.val)*np.sec(x.val) * x.dot)
    return df

def csc(x:FSym) -> FSym: 
    df = FSym(np.csc(x.val), 
            -np.cot(x.val)*np.csc(x.val) * x.dot)
    return df

def sinh(x:FSym) -> FSym:
    df = FSym(np.sinh(x.val), 
            np.cosh(x.val)*x.dot)
    return df

def cosh(x:FSym) -> FSym:
    df = FSym(np.cosh(x.val), 
            -np.sinh(x.val)*x.dot)
    return df

def tanh(x:FSym) -> FSym:
    df = FSym(np.tanh(x.val), 
            np.sech(x.val)**2 * x.dot)
    return df

def coth(x:FSym) -> FSym: 
    df = FSym(np.tanh(x.val), 
            -np.csch(x.val)**2 * x.dot)
    return df

def sech(x:FSym) -> FSym: 
    df = FSym(np.sech(x.val), 
            np.tanh(x.val)*np.sech(x.val) * x.dot)
    return df

def csch(x:FSym) -> FSym: 
    df = FSym(np.csch(x.val), 
            -np.coth(x.val)*np.csch(x.val) * x.dot)
    return df

def asin(x:FSym) -> FSym: ...
def acos(x:FSym) -> FSym: ...
def atan(x:FSym) -> FSym: ...
def acot(x:FSym) -> FSym: ...
def asec(x:FSym) -> FSym: ...
def acsc(x:FSym) -> FSym: ...

def asinh(x:FSym) -> FSym: ...
def acosh(x:FSym) -> FSym: ...
def atanh(x:FSym) -> FSym: ...
def acoth(x:FSym) -> FSym: ...
def asech(x:FSym) -> FSym: ...
def acsch(x:FSym) -> FSym: ...

if __name__ == '__main__':
    ... 
