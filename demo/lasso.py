import torch
import numpy as np
import plotly.graph_objects as graph
from main import GD, LASSO


T = int(1e3)
A = np.load("data/A.npy")
b = np.load("data/b.npy")

learning_rate_function = lambda s : 0.001 / (s + 1)
learning_rate_c = lambda s : 0.001

gd  = GD(LASSO(A,b), x_init=torch.zeros(100), epochs=T, learning_rate = learning_rate_function)
gdc = GD(LASSO(A,b), x_init=torch.zeros(100), epochs=T, learning_rate = learning_rate_c )
gd.go()
gdc.go()

# plot the data
all_history = { "decreasing step size" : gd.history ,
                "0.01 step size"       : gdc.history }

fig = graph.Figure(layout = graph.Layout(title=graph.layout.Title(text="Plot lr=0.1 for LASSO")))
for i in all_history:
    fig.add_trace(graph.Scatter(x    = all_history[i]["step"],
                                y    = all_history[i]["function_vals"],
                                name = i))
fig.show()


