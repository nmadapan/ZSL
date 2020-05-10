class Rectangle:
    def __init__(self, length = 2., width = 3.):
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

    def perimeter(self):
        return 2 * self.length + 2 * self.width

# Here we declare that the Square class inherits from the Rectangle class
class Square(Rectangle):
    def __init__(self, clamp = 3.):
        super().__init__()
        self.clamp = clamp

if __name__ == '__main__':
	sq = Square()
	print(sq.area())
	print(sq.perimeter())
	print(sq.length)
	print(sq.width)