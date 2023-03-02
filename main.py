# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# For data preprocess
import numpy as np
import pandas as pd
import csv
import os

tr_path = 'iris.csv'  # path to training data
tt_path = 'iris.csv'   # path to testing data
# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns


data_set = pd.read_csv("Iris.csv")
print(data_set.head())

X = data_set.iloc[:, :4]
y = data_set.iloc[:, 4]
from sklearn.preprocessing import normalize
X = normalize(X)
y = np.array(y)
y = y.astype(int)

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.1,random_state=0)
print('X_train shape: ',X_train.shape)
print('X_test shape: ',X_test.shape)
print('Y_train shape: ',Y_train.shape)
print('Y_test shape: ',Y_test.shape)

from torch.utils.data import TensorDataset, DataLoader
trainloader = DataLoader(TensorDataset(torch.from_numpy(X_train),torch.from_numpy(Y_train)),batch_size=135,shuffle=True)
testloader = DataLoader(TensorDataset(torch.from_numpy(X_test),torch.from_numpy(Y_test)),batch_size=135,shuffle=False)

dataloaders = {
    "train": trainloader,
    "validation": testloader

}

class Classifier(nn.Module):
    ''' A simple fully-connected deep neural network '''
    def __init__(self):
        super().__init__()
        # Define your neural network here
        # TODO: How to modify this model to achieve better performance?
        self.fc1 = nn.Linear(4, 227)
        self.fc2 = nn.Linear(227, 94)
        self.fc3 = nn.Linear(94, 75)
        self.fc4 = nn.Linear(75, 3)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x


model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

print(model)

def predict(model,inputs):
    output = model(inputs)
    return output.data.numpy().argmax(axis= 1)

from torch.autograd import Variable

loss1 = []
train_acc = []

Epoch = 2000
for epoch in range(Epoch):
    acc = 0

    for i, (features, labels) in enumerate(trainloader):
        features = Variable(features)
        labels = Variable(labels)

        #forward pass and backward pass
        optimizer.zero_grad()
        features = features.float()
        outputs = model(features)

        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        if (i+1) % len(trainloader) == 0:
            Ypred = predict(model, torch.from_numpy(X_train).float())
            acc = np.mean(Y_train == Ypred)

            train_acc1 = acc / len(trainloader)
            train_acc.append(train_acc1)
            loss1.append(loss.data)
            if epoch%5 == 0:
                print(f'Epoch [{epoch}/{Epoch}], Iter [%d] Loss: {loss.data} Training Accuracy: {train_acc1}')

np_loss = loss1[0].numpy()
for i in range(len(loss1)):
    np_loss=np.append(np_loss,loss1[i])

np_acc = train_acc[0]
for i in range(len(train_acc)):
    np_acc = np.append(np_acc,train_acc[i])

plt.plot(np_acc,color='blue')
plt.title("Training Accuracy")
plt.show()

plt.plot(np_loss,color='red',label='Training loss')
plt.title("Training Loss")
plt.legend()
plt.show()

Ypred = predict(model,torch.from_numpy(X_test).float())
acc = np.mean(Y_test == Ypred)
print('Test accracy: ', acc)

from sklearn.metrics import classification_report
target_names = ['data_set-setosa','data_set-versicolor','data_set-virginica']
print(classification_report(Y_test,Ypred,target_names=target_names))





########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
########################################################






myseed = 42069  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['test'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['test'], c='tab:cyan', label='test')
    plt.ylim(0.0, 1.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()


class IrisDataset(Dataset):
    ''' Dataset for loading and preprocessing the COVID19 dataset '''

    def __init__(self,
                 path,
                 mode='train',
                 target_only=False):
        self.mode = mode

        # Read data into numpy arrays
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, :].astype(float)

        if not target_only:
            feats = list(range(4))  # feature means input
        else:
            # TODO: Using 40 states & 2 tested_positive features (indices = 57 & 75)
            pass
        print("feats")
        print(feats)
        if mode == 'test':
            # Testing data
            target = data[:, -1]
            data = data[:, feats]

            indices = [i for i in range(len(data)) if i % 5 == 0]

            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])
        else:
            # Training data (train/dev sets)
            # data: 2700 x 94 (40 states + day 1 (18) + day 2 (18) + day 3 (18))
            target = data[:, -1]
            data = data[:, feats]

            # Splitting training data into train & dev sets
            # if mode == 'train':
            #     indices = [i for i in range(len(data)) if i % 10 != 0]
            # elif mode == 'dev':
            #     indices = [i for i in range(len(data)) if i % 10 == 0]
            indices = [i for i in range(len(data)) if i % 5 != 0]
            # Convert data into PyTorch tensors
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        # Normalize features (you may remove this part to see what will happen)
        self.data[:, :] = \
            (self.data[:, :] - self.data[:, :].mean(dim=0, keepdim=True)) \
            / self.data[:, :].std(dim=0, keepdim=True)

        self.dim = self.data.shape[1]

        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            return self.data[index], self.target[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)

"""
DataLoader
"""
def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = IrisDataset(path, mode=mode, target_only=target_only)  # Construct dataset
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)                            # Construct dataloader
    return dataloader


"""Deep Neural Network"""

class NeuralNet(nn.Module):
    ''' A simple fully-connected deep neural network '''
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()

        # Define your neural network here
        # TODO: How to modify this model to achieve better performance?
        self.net = nn.Sequential(
            nn.Linear(input_dim, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        ''' Calculate loss '''
        # TODO: you may implement L1/L2 regularization here
        return self.criterion(pred, target)

"""
Training
"""

def train(tr_set, dv_set, model, config, device):
    ''' DNN training '''

    n_epochs = config['n_epochs']  # Maximum number of epochs

    # Setup optimizer
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])

    min_mse = 1000.
    loss_record = {'train': [], 'test': []}      # for recording training loss
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()                           # set model to training mode
        for x, y in tr_set:                     # iterate through the dataloader
            optimizer.zero_grad()               # set gradient to zero
            x, y = x.to(device), y.to(device)   # move data to device (cpu/cuda)
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
            mse_loss.backward()                 # compute gradient (backpropagation)
            optimizer.step()                    # update model with optimizer
            loss_record['train'].append(mse_loss.detach().cpu().item())

        # After each epoch, test your model on the validation (development) set.
        dev_mse = dev(dv_set, model, device)

        if dev_mse < min_mse: ## minimum loss record update
            # Save model if your model improved
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                .format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['test'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            # Stop training if your model stops improving for "config['early_stop']" epochs.
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record

"""
Validation
"""
def dev(dv_set, model, device):
    model.eval()                                # set model to evalutation mode
    total_loss = 0
    #print(dv_set)
    for x, y in dv_set:                         # iterate through the dataloader
        x, y = x.to(device), y.to(device)       # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            # print("xxxx")
            # print(pred, y)
            mse_loss = model.cal_loss(pred, y)  # compute loss
            # print(mse_loss)
        total_loss += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
        # print(f"total_loss{total_loss}")
    total_loss = total_loss / len(dv_set.dataset)              # compute averaged loss
    # print(f"total_loss AVG {total_loss}")

    return total_loss

"""Testing"""

def test(tt_set, model, device):
    model.eval()                                # set model to evalutation mode
    preds = []
    for x in tt_set:                            # iterate through the dataloader
        x = x.to(device)                        # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            preds.append(pred.detach().cpu())   # collect prediction
    preds = torch.cat(preds, dim=0).numpy()     # concatenate all predictions and convert to a numpy array
    return preds

"""
Setup Hyper-parameters
"""
device = get_device()                 # get the current available device ('cpu' or 'cuda')
os.makedirs('models', exist_ok=True)  # The trained model will be saved to ./models/
target_only = False                   # TODO: Using 40 states & 2 tested_positive features

# TODO: How to tune these hyper-parameters to improve your model's performance?
config = {
    'n_epochs': 500,                # maximum number of epochs
    'batch_size': 20,               # mini-batch size for dataloader
    'optimizer': 'SGD',              # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.001,                 # learning rate of SGD
        'momentum': 0.9              # momentum for SGD
    },
    'early_stop': 100,               # early stopping epochs (the number epochs since your model's last improvement)
    'save_path': 'models/model.pth'  # your model will be saved here
}

"""Load data and model"""
tr_set = prep_dataloader(tr_path, 'train', config['batch_size'], target_only=target_only)
tt_set = prep_dataloader(tt_path, 'test', config['batch_size'], target_only=target_only)
#print(f" dddiiimmm? {tr_set.dataset.dim}")
print(f" dataset {tr_set}")
for i in tr_set.dataset:  # Normalized features
    print(i)
model = NeuralNet(tr_set.dataset.dim).to(device)  # Construct model and move to device
print(f"tr_set: {tr_set}")


model_loss, model_loss_record = train(tr_set, tt_set, model, config, device)
print("print parameter")
for param in model.parameters():
    print("****")
    print(param)
plot_learning_curve(model_loss_record, title='deep model')


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
