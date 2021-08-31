import cv2
import numpy as np

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def x(self):
        return self._x

    def y(self):
        return self._y

    def norm(self):
        return cv2.norm((self.x, self.y))

    def sqrt(self):
        return Point(np.sqrt(self.x), np.sqrt(self.y))

    def tup(self):
        return self.x, self.y

    def __add__(self, other):
        if type(other) == type(self):
            return Point(self.x + other.x, self.y + other.y)
        else:
            return Point(self.x + other, self.y + other)

    def __sub__(self, other):
        if type(other) == type(self):
            return Point(self.x - other.x, self.y - other.y)
        else:
            return Point(self.x - other, self.y - other)

    def __mul__(self, other):
        if type(other) == type(self):
            return Point(self.x * other.x, self.y * other.y)
        else:
            return Point(self.x * other, self.y * other)

    def __truediv__(self, other):
        if type(other) == type(self):
            return Point(self.x / other.x, self.y / other.y)
        else:
            return Point(self.x / other, self.y / other)

    def __lt__(self, other):
        return self.x < other.x and self.y < other.y

    def __gt__(self, other):
        return self.x > other.x and self.y > other.y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return "Point(" + str(self.x) + ", " + str(self.y) +  ")"