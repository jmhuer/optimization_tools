import torch


class GD:
    def __init__(self, function, x_init, epochs, learning_rate):
        self.function = function
        self.x_init = x_init
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.history = {"step": [],"function_vals": [],"grad_vals": [], "x_vals": [] }
    def go(self):
        x = torch.Tensor(self.x_init)
        for i in range(self.epochs):
          ##EVALUATE HERE
          y = self.function(x)
          g = self.function.gradient(x)
          ##UPDATE HERE : if lr rate is lamda function act different
          if isinstance(self.learning_rate, float):  x = x - self.learning_rate * g
          else:  x = x - self.learning_rate(i) * g
          ##STORE HISTORY
          self.history['step'].append(i)
          self.history['function_vals'].append(float(y))
          self.history['grad_vals'].append(g)
          self.history['x_vals'].append(x)




class AGD:
    def __init__(self, function, x_init, epochs, learning_rate):
        self.function = function
        self.momentum = lambda k : (k-1)/(k+2)
        self.x_init = x_init
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.history = {"step": [],"function_vals": [],"grad_vals": [], "x_vals": [] }
    def go(self):
        x = torch.Tensor(self.x_init)
        y = x
        for i in range(self.epochs):
          ##EVALUATE HERE
          f = self.function(x)
          if i!=0: y = x + self.momentum(i) * (x - self.history['x_vals'][-1])
          g = self.function.gradient(y)
          ##UPDATE HERE : if lr rate is lamda function act different
          x = y - self.learning_rate(i) * g[0]
          ##STORE HISTORY
          self.history['step'].append(i)
          self.history['function_vals'].append(float(f))
          self.history['grad_vals'].append(g)
          self.history['x_vals'].append(x)

