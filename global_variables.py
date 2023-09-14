RED = 0
BLUE = 1
square = 1  # bottom_right
angle = 2


class MySize:
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def get(self):
        return self.w, self.h

    def copy(self):
        obj = MySize(self.w, self.h)
        return obj

    def __getitem__(self, item):
        if item == 0:
            return self.w
        elif item == 1:
            return self.h
        else:
            return None

    def __lt__(self, other):
        return self.w < other.w and self.h < other.h

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __eq__(self, other):
        return self.w == other.w and self.h == other.h

    def __ne__(self, other):
        pass

    def __gt__(self, other):
        return self.w > other.w and self.h > other.h

    def __ge__(self, other):
        pass
