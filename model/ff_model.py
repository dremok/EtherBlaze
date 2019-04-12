from collections import OrderedDict

import torch
from torch import nn, optim

from model.base_model import BaseModel


class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class FeedForwardClassificationModel(BaseModel):
    def __init__(self, num_classes):
        self._simple_net = SimpleNet(784, 512, num_classes)
        self._optimizer = optim.SGD(self._simple_net.parameters(), lr=0.01, momentum=0.5)
        modules = OrderedDict([('mnist', self._simple_net)])
        super().__init__(modules)

    def train(self, train_loader):
        self._simple_net.train()
        # TODO: Finish this method...
        # for batch_idx, (data, target) in enumerate(train_loader):
        #     data, target = data.to(device), target.to(device)
        #     optimizer.zero_grad()
        #     output = model(data)
        #     loss = F.nll_loss(output, target)
        #     loss.backward()
        #     optimizer.step()
        #     if batch_idx % args.log_interval == 0:
        #         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #             epoch, batch_idx * len(data), len(train_loader.dataset),
        #                    100. * batch_idx / len(train_loader), loss.item()))

    def predict(self, input_data):
        with torch.no_grad():
            outputs = self._simple_net(input_data.float())
            _, predicted = torch.max(outputs.data, 0)
            return predicted.item()
