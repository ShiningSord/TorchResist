import numpy as np
# import hashlib as hl
from abc import ABC
from typing import List, Tuple, Union

class Point:
    def __init__(self, x : int, y : int) -> None:
        self._x = x
        self._y = y
        # self.md5 = hl.md5()
        # self.md5.update(bytes(self.__str__(), 'utf-8'))
    
    def __str__(self) -> str:
        return f"({self._x}, {self._y})"
    
    def x(self):
        return self._x

    def y(self):
        return self._y
    
    def list(self):
        return [self._x, self._y]
    
    def __hash__(self) -> int:
        return hash((self.x, self.y))
    
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Point):
            return False
        else:
            return __value.x() == self.x() and __value.y() == self.y()
    
    def numpy(self):
        return np.array(self.list())
    
    def __add__(self, shift : Union[List[int], Tuple[int]]):
        assert isinstance(shift, (List, Tuple)) and len(shift) == 2
        return Point(self.x() + shift[0], self.y() + shift[1])
    
    def __mul__(self, size : int):
        assert isinstance(size, int) and size > 0
        return Point(self.x() * size, self.y() * size)
    
    def __sub__(self, shift):
        assert isinstance(shift, (List, Tuple)) and len(shift) == 2
        return Point(self.x() - shift[0], self.y() - shift[1])
         
class Line(ABC):
    directions = ["left", "right", "up", "down"]
    def __init__(self) -> None:
        super().__init__()
        
    def fixAxis(self):
        raise not NotImplementedError()
    
    def mutableStartPoint(self):
        raise not NotImplementedError()
    
    def shift(self):
        raise not NotImplementedError()
    
    def inLine(self, p : Point):
        raise not NotImplementedError()
    
    def __str__(self) -> str:
        raise not NotImplementedError()
        

class VerticalLine(Line):
    def __init__(self, p1 : Point, p2 : Point) -> None:
        assert p1.x() == p2.x(), f"Points need to have same index in x axis."
        self.fix = p1.x()
        super().__init__()
        self.p1, self.p2 = p1, p2
    
    def fixAxis(self):
        return self.fix
    
    def mutableStartPoint(self):
        return self.p1.y()
    
    def shift(self):
        return self.p2.y() - self.p1.y()
    
    def inLine(self, p: Point):
        if p.x() == self.fixAxis():
            start = min(self.p1.y(), self.p2.y())
            end = max(self.p1.y(), self.p2.y())
            if p.y() <= end and p.y() >= start:
                return True
            return False
        return False
    
    def __str__(self) -> str:
        return f"{self.p1} T {self.p2}"
    
class HorizontalLine(Line):
    def __init__(self, p1 : Point, p2 : Point) -> None:
        assert p1.y() == p2.y(), f"Points need to have same index in y axis."
        self.fix = p1.y()
        super().__init__()
        self.p1, self.p2 = p1, p2
        
    def fixAxis(self):
        return self.fix
        
    def mutableStartPoint(self):
        return self.p1.x()    
        
    def shift(self):
        return self.p2.x() - self.p1.x()
    
    def inLine(self, p: Point):
        if p.y() == self.fixAxis():
            start = min(self.p1.x(), self.p2.x())
            end = max(self.p1.x(), self.p2.x())
            if p.x() <= end and p.x() >= start:
                return True
            return False
        return False

    def __str__(self) -> str:
        return f"{self.p1} -> {self.p2}"
    
def autoLine(p1 : Point, p2 : Point):
    if p1.x() == p2.x():
        return VerticalLine(p1, p2)
    elif p1.y() == p2.y():
        return HorizontalLine(p1, p2)
    else:
        raise ValueError(f"Can't construct line between {p1} and {p2}")

class BBox:
    def __init__(self, ll : Point, ur : Point) -> None:
        self.ll : Point = ll
        self.ur : Point = ur
        # self.md5 = hl.md5()
        # self.md5.update(bytes(self.__str__(), 'utf-8'))

    def __str__(self) -> str:
        s = ", ".join([str(_) for _ in self.list()])
        return f"({s})"
    
    def list(self):
        return [self.ll.x(), self.ll.y(), self.ur.x(), self.ur.y()]
    
    def getCornerPoints(self):
        return [self.ll, 
                Point(self.ll.x(), self.ll.y() + self.getHeight()), 
                self.ur, 
                Point(self.ur.x(), self.ur.y() - self.getHeight())]
    
    def __hash__(self) -> int:
        return hash((self.ll, self.ur))
    
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, BBox):
            return False
        return self.ll == __value.ll and self.ur == __value.ur
    
    def getWidth(self):
        return abs(self.ur.x() - self.ll.x())
    
    def getHeight(self):
        return abs(self.ur.y() - self.ll.y())
    
    def getLowLeft(self):
        return self.ll
    
    def getUpRight(self):
        return self.ur
    
    def getUpLeft(self):
        return Point(self.ll.x(), self.ur.y())
    
    def getLowRight(self):
        return Point(self.ur.x(), self.ll.y())
    
    def isBoundary(self, p : Point):
        corners = self.getCornerPoints()
        lines: List[Line] = [
            autoLine(corners[0], corners[1]),
            autoLine(corners[1], corners[2]),
            autoLine(corners[2], corners[3]),
            autoLine(corners[3], corners[0]),
        ] 
        
        for line in lines:
            if line.inLine(p):
                return True
        return False
        
    def isInside(self, p : Point, bound=True):
        if bound:
            if p.x() <= self.ur.x() and p.x() >= self.ll.x() and p.y() <= self.ur.y() and p.y() >= self.ll.y():
                return True
            return False
        else:
            if p.x() < self.ur.x() and p.x() > self.ll.x() and p.y() < self.ur.y() and p.y() > self.ll.y():
                return True
            return False
    
    def isCorner(self, p : Point):
        for c in self.getCornerPoints():
            if p == c:
                return True
        return False
        
    def numpy(self, fold=False):
        if fold:
            return np.vstack((self.ll.numpy(), self.ur.numpy()))
        return np.hstack((self.ll.numpy(), self.ur.numpy()))

