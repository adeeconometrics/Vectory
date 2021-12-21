from __future__ import annotations
from typing import List, Union
from numbers import Real
from math import sqrt, acos

class vector:
    """
    Representation of real-valued vector :math:`\\vec{v} \\in \\mathbb{R}^n`
    """

    contents: List[Real]

    def __init__(self, vector:list) -> None: 
        self.contents = vector
        if any(not isinstance(x,Real) for x in self.contents):
            raise TypeError('Vector is expecting a real-valued element')

        self.__current: int = -1
        self.max = len(self.contents)

    def __str__(self) -> str:
        return f'{self.contents}'

    def __repr__(self) -> str: 
        return f'{self.contents}'

    def __add__(self, other:vector) -> vector: 
        """implementation of vector addition :math:`\\vec{v_1} + \\vec{v2}`

        Args:
            other (vector): represents the addend vector :math:`\\vec{v_2}`

        Raises:
            TypeError: when other is not of type vector
            ValueError: when other is not the same size as the vector

        Returns:
            vector: result of element-wise addition of :math:`\\vec{v_1}` and :math:`\\vec{v_2}`
        """        
        if not type(other) is type(self):
            raise TypeError('Type is expected to be a vector')
        if len(other) != len(self.contents):
            raise ValueError('Vector is expected to be of equal size')

        return vector([x0 + x1 for (x0,x1) in zip(self.contents, other.contents)])

    def __sub__(self, other:vector) -> vector: 
        """implementation of vector subtraction :math: `\\vec{v_1} - \\vec{v2}`

        Args:
            other (vector): represents the subtrahend vector :math:`\\vec{v_2}`

        Raises:
            TypeError: when other is not of type vector
            ValueError: when other is not the same size as the vector

        Returns:
            vector: result of element-wise subtraction of :math:`\\vec{v_1}` and :math:`\\vec{v_2}`
        """        
        if not type(other) is type(self):
            raise TypeError('Type is expected to be a vector')
        if len(other) != len(self.contents):
            raise ValueError('Vector is expected to be of equal size')
            
        return vector([x0 - x1 for (x0,x1) in zip(self.contents, other.contents)])

    def __mul__(self, other:Union[vector, Real]) -> Union[vector, Real]: 
        """implementation of vector element-wise multiplication :math: `\\vec{v_1} * \\vec{v2}`
        and scalar-vector multiplication :math:`c\\vec{v} \\| \\ c\\in \\mathbb{R}`

        Args:
            other (vector | Real): represents the addend vector :math:`\\vec{v_2}`

        Raises:
            TypeError: when other is not of type vector or instance of Real
            ValueError: when other is not the same size as the vector

        Returns:
            vector: result of element-wise multiplication of :math:`\\vec{v_1}` and :math:`\\vec{v_2}`
        """
        if isinstance(other, Real):
            return vector([other* x0 for x0 in self.contents])
        if not type(other) is type(self):
            raise TypeError('Type is expected to be a vector')
        if len(other) != len(self.contents):
            raise ValueError('Vector is expected to be of equal size')

        return vector([x0 * x1 for (x0,x1) in zip(self.contents, other.contents)])

    def __truediv__(self, other:Union[vector, Real]) -> Union[vector, Real]: 
        """implementation of vector element-wise division :math: `\\vec{v_1} / \\vec{v2}`
        and scalar-vector division :math:`\\vec{v}/c \\| \\ c\\in \\mathbb{R}`

        Args:
            other (vector | Real): represents the addend vector :math:`\\vec{v_2}`

        Raises:
            TypeError: when other is not of type vector or instance of Real
            ValueError: when other is not the same size as the vector

        Returns:
            vector: result of element-wise addition of :math:`\\vec{v_1}` and :math:`\\vec{v_2}`
        """
        if isinstance(other, Real):
            return vector([x0/other for x0 in self.contents])
        if not type(other) is type(self):
            raise TypeError('Type is expected to be a vector')
        if len(other) != len(self.contents):
            raise ValueError('Vector is expected to be of equal size')

        return vector([x0/x1 for (x0,x1) in zip(self.contents, other.contents)])

    def __pow__(self, other:Real) -> vector:
        """
        Args:
            other (Real): exponent to which the vector is raised

        Raises:
            TypeError: when other is not an instance of Real

        Returns:
            vector: returns a vector with elements raised to `other`
        """        
        if isinstance(other, Real):
            return vector([x0**other for x0 in self.contents])
        raise TypeError('Types are expected to an instance of be Real')

    def __eq__(self, other) -> bool:
        return all(x == y for (x,y) in zip(self.contents, other.contents))

    def __ne__(self, other) -> bool: 
        return any(x != y for (x,y) in zip(self.contents, other.contents))

    def __iter__(self): 
        return self

    def __next__(self): 
        self.__current += 1
        if self.__current < self.max:
            return self.contents[self.__current]
        raise StopIteration
    
    def __len__(self) -> int:
        return len(self.contents)
    
    def __getitem__(self, index:int):
        if index < self.max:
            return self.contents[index]
        raise IndexError('invalid index')

def norm(x:vector) -> float:
    """
    .. math::
        ||\\vec{x}|| := \\sqrt{x^2_1 + ... + x^2_n}

    Args:
        x (vector): represents the coordinates of a vector

    Raises:
        TypeError: when x is not of type vector

    Returns:
        float: norm of a vector :math:`||\\vec{x}||`
    """    
    if type(x) is not vector:
        raise TypeError('type expects a vector')
    return sqrt(sum(x**2 for x in x.contents))

    
def normalize(p:vector, q:vector) -> vector:
    """
    Args:
        p (vector): repreesents the coordinates of a vector
        q (vector): represents the coordinates of a vector

    Raises:
        TypeError: when p and q are not of type vector

    Returns:
        vector: coordinates of normalized vector
    """    
    if type(p) is not vector or type(q) is not vector:
        raise TypeError('p and q are expected to be a vector')
    
    dist:vector = p-q
    return (dist)/norm(dist)

def dot(p:vector, q:vector) -> float:
    """dot product operation of two vectors

    .. math::
        \\vec{p} \\cdot \\vec{q} = \\sum_{i=1}^n p_i q_i

    Args:
        p (vector): represents the coordinates of vector p
        q (vector): represents the coordinates of vector q

    Raises:
        ValueError: when p and q are not the same in lenght

    Returns:
        float: dot product of vector p and q 
    """    
    if len(p) != len(q):
        raise ValueError('lengths of vectors must be equal')
    return sum(x0*x1 for (x0,x1) in zip(p,q))

def cross(p:vector, q:vector) -> vector:
    """
    A cross product multiplication for vectors in :math:`\\mathbb{R}^3`
    
    .. math::
        \\vec{p} \\times \\vec{q}

    Args:
        p (vector): represents coordinates of vector p
        q (vector): represents coordinates oF vector q

    Raises:
        NotImplementedError: [description]

    Returns:
        vector: [description]
    """    
    if len(p) != 3 and len(q) != 3:
        raise NotImplementedError()
    
    result = []
    result.append(p[1]*q[2] - p[2]*q[1])
    result.append(p[2]*q[0] - p[0]*q[2])
    result.append(p[0]*q[1] - p[1]*q[0])
    return vector(result)


def get_angle(p:vector, q:vector) -> float: 
    """
    ..math::
        \\theta = \\acos \\left( \\frac{\\vec{p} \\cdot \\vec{q}}{|| \\vec{p} || || \\vec{q} ||}\\right)

    Args:
        p (vector): represents coordinates of vector p
        q (vector): represents coordinates of vector q

    Returns:
        float: angle between p and q
    """    
    return acos(dot(p,q)/(norm(p)*norm(q)))

def get_projection(p:vector, q:vector) -> vector: 
    """
    .. math:: 
        \\text{projection}_{\\vec{p}} \\vec{q} = \\frac{(\\vec{p} \\cdot \\vec{q})}{|| \\vec{p} ||^2} \\vec{p}

    Args:
        p (vector): represents coordinates of p vector
        q (vector): represents coordinates of q vector

    Returns:
        vector: coordinates of the projection of p to q
    """    
    return p * (dot(p,q)/norm(p)**2 )

def dist(p:vector, q:vector) -> vector:
    """
    Args:
        p (vector): represents coordinates of p vector
        q (vector): represents coordinates of q vector

    Raises:
        TypeError: when either p or q are not of type vector

    Returns:
        vector: returns the distance of p to q
    """    
    if type(p) is not vector or type(q) is not vector:
        raise TypeError('p and q are expected to be of type vector')    
    return p - q

def vectorize() -> vector: ...

if __name__ == '__main__':
    # v0 = vector([1,2,4,5])
    # v1 = vector([1,2,3,5])

    # print(f'v0 = {v0}')
    # print(f'v1 = {v1}')
    # print(f'length: {len(v0)}')
    # print(f'add: {v0 + v1}')
    # print(f'sub: {v0 - v1}')
    # print(f'mul: {v0 * v1}')
    # print(f'mul2: {v1*2}')
    # print(f'div: {v1/2}')
    # print(f'div2: {v0/v1}')
    # print(f'pow: {v1**2}')
    # print(f'eq: {v1 == vector([1,2,3,5])}')
    # print(f'neq: {v1 != vector([1,3,4,5])}')

    from fractions import Fraction as F
    p = vector([F(3),F(-1),F(1)])
    q = vector([F(0),F(5),F(-1)])
    print(normalize(p,q))