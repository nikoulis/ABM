class Base:
    def __init__(self, list=[]):
        self.list = list

class Sub(Base):
    def __init__(self, item):
        Base.__init__(self)
        self.list.append(item)

x = Sub(1)
y = Sub(2)

print x.list
print y.list
