import torch
from torch import matmul
from torch.autograd.functional import hessian
from torch.autograd.functional import jacobian
import numpy as np
import pickle

class Data:
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y

file= open("data/logistic_news/train.obj",'rb')
train = pickle.load(file)
file.close()


file = open("data/logistic_news/test.obj",'rb')
test = pickle.load(file)
file.close()



class Logistic_loss(torch.nn.Module):
    def __init__(self, data, init_weights, mou=0.001):
        super(Logistic_loss, self).__init__()
        self.X = torch.Tensor(data.X)
        self.Y = torch.Tensor(data.Y)
        self.N = self.X.size()[0]
        self.classes = init_weights.size()[0]
        self.mou = mou
        self.weights = init_weights
        self.optimal = 0

    def forward(self, weights):
        exponent_dots = lambda i: -torch.sum(self.X[i, :] * weights, dim=-1)
        inside_sum = lambda i: matmul(weights[self.Y[i].int(), :], self.X[i, :]) + torch.logsumexp(exponent_dots(i), 0)
        return (1 / self.N) * sum(inside_sum(i) for i in range(self.N)) + self.mou * (torch.norm(weights, 2) ** 2)

    def gradient(self, x):
        return jacobian(self.forward, inputs=torch.Tensor(x), create_graph=True)

    def test(self, testset, weights):
        total_labels = torch.tensor(testset.Y).size()[0]
        _dots = lambda i: torch.sum(torch.tensor(testset.X[i, :]) * weights, dim=-1)
        labels = sum(torch.equal(torch.tensor(testset.Y[i]), torch.argmin(_dots(i))) for i in range(total_labels))
        print("correctly labeled: ", labels)
        return labels / total_labels
    def test_loss(self, testset, weights):
        X = torch.tensor(testset.X).float()
        Y = torch.tensor(testset.Y).float()
        N = X.size()[0]
        exponent_dots = lambda i: -torch.sum(X[i, :] * weights, dim=-1)
        inside_sum = lambda i: matmul(weights[Y[i].int(), :], X[i, :]) + torch.logsumexp(exponent_dots(i), 0)
        return round(float((1 / N) * sum(inside_sum(i) for i in range(N)) + self.mou * (torch.norm(weights, 2) ** 2)),2)
    def manual_gradient(self, weights):  ##optionally here is the manual calculation of grad
        n, d = self.X.shape
        X, Y = np.array(self.X), np.array(self.Y)
        g_loss = (1. / Y.shape[0]) * np.dot(X.T, -Y * (1. / (1. + np.exp(np.dot(X, self.weights) * Y))))
        g_reg = 2 * 0.0001 * self.weights
        return torch.tensor(g_loss + g_reg)


# manual calc to verify function
weights = torch.tensor([[1, 3, 1],
                        [2, 1, 0]]).float()

x = torch.tensor([[1, 0, 2],
                  [0, 1, 1],
                  [1, 2, 0]]).float()


y = torch.tensor([0, 1, 0]).float()
d = Data(x,y)

c = Logistic_loss(data=d, init_weights=weights)

print(c.manual_gradient(weights))
print(c.gradient(weights))
exit()
init_weights = torch.rand(20, 2001)

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = Logistic_loss(data=train,init_weights=init_weights)
learning_rate_function = lambda s : 5

gd  = AGD(model, x_init = init_weights, epochs=30, learning_rate = learning_rate_function, test_set = test)
gd2 =  GD(model, x_init = init_weights, epochs=30, learning_rate = learning_rate_function, test_set = test)
gd.go()
gd2.go()
# plot the data
all_history = { "AGD": gd.history,
                "GD" : gd2.history}

for r in ("function_vals","test_loss"):
  fig = graph.Figure(layout = graph.Layout(title=graph.layout.Title(text= r + " logistic loss reg")))
  for i in all_history:
      fig.add_trace(graph.Scatter(x    = all_history[i]["step"],
                                  y    = all_history[i][r],
                                  name = r + " "+ i))
  fig.show()


