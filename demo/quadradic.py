import numpy as np
from main import GD, Quadratic2
import plotly.graph_objects as graph

Q1 = np.array([[ 1.17091573, -0.03686123, -0.1001259 ],
       [-0.03686123,  1.03835691,  0.17285956],
       [-0.1001259 ,  0.17285956,  1.06072736]])
Q2 = np.array([[ 15.27095759,  -1.97915834, -14.22190153],
       [ -1.97915834,   0.34660759,   1.91586927],
       [-14.22190153,   1.91586927,  15.76943482]])
Q3 = np.array([[28.59657006,  0.3684004 ,  0.90750259],
       [ 0.3684004 , 28.11480924,  0.81866989],
       [ 0.90750259,  0.81866989, 28.7886207 ]])
q1 = np.array([-4.68686663, -0.89027884, -1.57340281])
q2 = np.array([ 6.75973216,  1.23223936, -0.87956749])
q3 = np.array([ 0.8684369 , -4.69970837, -1.09690726])
c1 = 1.61888566;
c2 = -2.66426693;
c3 = 0.84184738;

'''
Here we run and plot all 3 functions
'''

learning_rate = [lambda x: 0.01,lambda x: 0.005, lambda x:0.0001]

functions = [Quadratic2(Q1, q1, c1) ,
             Quadratic2(Q2, q2, c2) ,
             Quadratic2(Q3, q3, c3)]

plot_titles  = ['Q1,q1,c1 plot',
                'Q2,q2,c2 plot',
                'Q3,q3,c3 plot']

for (f, title) in zip(functions, plot_titles):
  gda = GD(f, x_init=[0,0,0], epochs=100, learning_rate=learning_rate[0])
  gdb = GD(f, x_init=[0,0,0], epochs=100, learning_rate=learning_rate[1])
  gdc = GD(f, x_init=[0,0,0], epochs=100, learning_rate=learning_rate[2])
  gda.go()
  gdb.go()
  gdc.go()

  # plot the data
  all_history = {learning_rate[0] : gda.history,
                 learning_rate[1] : gdb.history,
                 learning_rate[2] : gdc.history}

  fig = graph.Figure(layout=graph.Layout(title=graph.layout.Title(text=title)))
  for i in all_history:
      fig.add_trace(graph.Scatter(x    = all_history[i]["step"],
                                  y    = all_history[i]["function_vals"],
                                  name = i))
  fig.show()



