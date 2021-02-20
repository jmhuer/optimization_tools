import pickle

class Data:
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y

class CustomUnpickler(pickle.Unpickler):
  def find_class(self, module, name):
    try:
      return super().find_class(__name__, name)
    except AttributeError:
      return super().find_class(module, name)
