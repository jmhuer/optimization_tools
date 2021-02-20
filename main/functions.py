import torch
from torch import matmul
from torch.autograd.functional import hessian
from torch.autograd.functional import jacobian
import numpy as np


class LASSO(torch.nn.Module):
    def __init__(self, A, b, lmda = 0.2):
        super(LASSO, self).__init__()
        self.x = torch.nn.Parameter(torch.randn(np.shape(A)[1]))
        self.A = torch.Tensor(A)
        self.b = torch.Tensor(b)
        self.lmda = lmda
        self.unsquared_error = lambda x : torch.sum((matmul(self.A, x) - self.b)**2).sqrt()
        self.reg = lambda x : self.lmda * torch.linalg.norm(((self.x)), 1, -1)
        self.function = lambda x : (1/2) * self.unsquared_error(x)**2 + self.reg(x)
    def forward(self, x):
        return self.function(torch.Tensor(x))
    def gradient(self,x):
        return jacobian(self.function, inputs=torch.Tensor(x))
    def hessian(self,x):
        return hessian(self.function, inputs=torch.Tensor(x))
    # def subgradient(self,x):
    #     self.x = torch.nn.Parameter(torch.Tensor(x))
    #     self.function(self.x).backward() #pytorch defines grad=0 at x=0 which is the one subgradient
    #     return self.x.grad

class Quadratic(torch.nn.Module):
    def __init__(self, a, b, c, input_size = 3):
        super(Quadratic, self).__init__()
        self.x = torch.nn.Parameter(torch.randn(input_size))
        self.a = torch.Tensor(a)
        self.b = torch.Tensor(b)
        self.c = torch.Tensor([c])
        self.function = lambda x : (1/2) * matmul(matmul(x ,self.a), x) + matmul(self.b, x) + self.c
    def forward(self, x):
        self.x = torch.nn.Parameter(torch.Tensor(x))
        return self.function(self.x)
    def gradient(self,x):
        self.x = torch.nn.Parameter(torch.Tensor(x))
        self.function(self.x).backward(gradient = torch.tensor([1])) #note gradient could be implicintly passed
        return self.x.grad


class Logistic_loss(torch.nn.Module):
    def __init__(self, data, init_weights=torch.rand(20, 2001)):
        super(Logistic_loss, self).__init__()
        self.X = torch.Tensor(data.X)
        self.Y = torch.Tensor(data.Y)
        self.N = self.X.size()[0]
        print("N data points: ", self.N )
        self.weights = init_weights
    def forward(self, weights):
        exponent_dots = lambda i : -torch.sum(self.X[i,:] * weights, dim=-1)
        inside_sum = lambda i : matmul(weights[self.Y[i].int(),:], self.X[i,:]) + torch.logsumexp(exponent_dots(i), 0)
        return (1/self.N) * sum(inside_sum(i) for i in range(self.N)) + 0.001*torch.norm(self.weights,2)
    def gradient(self,x):
        return jacobian(self.forward, inputs=torch.Tensor(x))
    def step_size_calculator(self,x):
        hessian = self.hessian(x)
        min_eig = torch.eig(hessian)[0][:,0].min()
        max_eig = torch.eig(hessian)[0][:,0].max()
        return float(1/max_eig)
    def hessian(self,x):
        return hessian(self.forward, inputs=torch.Tensor(x))
    def test(self,testset, weights):
        total_labels = torch.tensor(testset.Y).size()[0]
        _dots = lambda i : torch.sum(torch.tensor(testset.X[i,:]) * weights, dim=-1)
        labels = sum(torch.equal(torch.tensor(testset.Y[i]), torch.argmin(_dots(i))) for i in range(total_labels))
        print(labels)
        return labels/total_labels


class Quadratic2(torch.nn.Module):
    def __init__(self, a, b, c, input_size = 3):
        super(Quadratic2, self).__init__()
        self.a = torch.Tensor(a)
        self.b = torch.Tensor(b)
        self.c = torch.Tensor([c])
        self.function = lambda x : (1/2) * matmul(matmul(x ,self.a), x) + matmul(self.b, x) + self.c
    def forward(self, x):
        return self.function(torch.Tensor(x))
    def gradient(self,x):
        return jacobian(self.function, inputs=torch.Tensor(x))
    def hessian(self,x):
        return hessian(self.function, inputs=torch.Tensor(x))




class Multivar_Costum(torch.nn.Module):
    def __init__(self, function, input_size = 3):
        super(Multivar_Costum, self).__init__()
        self.input_size = input_size
        self.a = torch.Tensor(a)
        self.b = torch.Tensor(b)
        self.c = torch.Tensor([c])
        self.function = lambda x: (1 / 2) * matmul(matmul(x, self.a), x) + matmul(self.b, x) + self.c
    def forward(self, x):
        return self.function(torch.Tensor(x))
    def gradient(self, x):
        return jacobian(self.function, inputs=torch.Tensor(x))
    def hessian(self, x):
        return hessian(self.function, inputs=torch.Tensor(x))
    def hessian_condition_number(self, x):
        min_eig = torch.max(torch.eig(function.hessian(x_init))[0][0:self.input_size, 0])
        max_eig = torch.min(torch.eig(function.hessian(x_init))[0][0:self.input_size, 0])
        return min_eig, max_eig