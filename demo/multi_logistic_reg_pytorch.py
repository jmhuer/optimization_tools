import torch.optim as optim
import torch
from main import CustomUnpickler

##this is multinomional logistic regression (optmized using cross entropy = maximum liklyhood)
def evaluate(model,testX,testY):
  total_labels = torch.tensor(testY).size()[0]
  _dots = lambda i : model(testX[i,:])
  labels = sum(torch.equal(torch.tensor(testY[i]), torch.argmax(_dots(i))) for i in range(total_labels))
  return labels/total_labels, labels

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using device: ", device)

##"Load data"
train = CustomUnpickler(open("data/logistic_news/train.obj",'rb')).load()
test = CustomUnpickler(open("data/logistic_news/test.obj",'rb')).load()

X,Y = torch.from_numpy(train.X).to(device), torch.from_numpy(train.Y).to(device)
testX,testY = torch.from_numpy(test.X).to(device), torch.from_numpy(test.Y).to(device)

model = linear = torch.nn.Linear(2001,20).double().to(device)

loss_fn = torch.nn.CrossEntropyLoss().to(device)

optimizer = optim.RMSprop(model.parameters(),lr=.05, weight_decay = 0.00001, momentum=0.05) ##this has weight decay just like you implemented
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

epochs = 10000
for i in range(epochs):
  optimizer.zero_grad()
  output = model(X)
  loss = loss_fn(output, Y)
  loss.backward()
  optimizer.step()
  percent, correct = evaluate(model,testX,testY)
  print("Epoch : {} \t Loss : {} \t correct_labels : {} \t precision: {}".format(i, round(float(loss),4), correct, round(float(percent),4)))

