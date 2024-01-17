class person:
  def __new__(cls, name):
    print("new: ", cls)
    print("new2: ", super())
    print(help(super))
    return super().__new__(cls)

  def __init__(self, name):
    print("init: ", name)
    self.name = name


person('x')
