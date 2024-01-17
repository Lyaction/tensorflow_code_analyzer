import six

class A(type):
  a=1
  def __new__(cls, name, bases, attrs):
    print('new: ')
    obj = super().__new__(cls, name, bases, attrs)
    print('new obj', obj)
    return obj

  def __call__(cls, *args):
    print('call: ', args, cls)
    obj = cls.__new__(cls)
    print('call: ', obj)
    obj.__init__(args)
    print('call super:', super(A, cls).__class__)
    #return super(A, cls).__call__('y')
    return obj


#x = six.with_metaclass(A)

class B(six.with_metaclass(A)):
  def __init__(self, name):
    self.name = name
    print("init: done")


y=B('x')
#print(dir(y), y.name, B.__mro__)
