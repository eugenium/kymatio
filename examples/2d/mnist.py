"""
Classification of MNIST with Scattering and Scattering hybrids
=====================================================================

Here we demonstrate a simple application of scattering on the MNIST dataset. Three models are demoed:
* linear - scattering + linear model
* mlp - scattering + MLP
* cnn - scattering + CNN

In all cases scattering features are normalized by batch normalization.

A similar example is given for cifar in the repository under exmpales/2d/cifar.py as an executable script with command line arguments

"""


###############################################################################
# If a GPU is available, let's use it!
import torch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

###############################################################################
# For reproducibility, we fix the seed of the random number generator.
torch.manual_seed(42)


###############################################
# Create dataloaders
from torchvision import datasets, transforms
import kymatio.datasets as scattering_datasets

if use_cuda:
    num_workers = 4
    pin_memory = True
else:
    num_workers = None
    pin_memory = False

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(scattering_datasets.get_dataset_dir('MNIST'), train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(scattering_datasets.get_dataset_dir('MNIST'), train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)


###############################################################################
# This will help us define networks a bit more cleanly
import torch.nn as nn
class View(nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1,*self.shape)


###############################################################################
# Create a training and test function
import torch.nn.functional as F

def train(model, device, train_loader, optimizer, scat):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(scat(data))
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()


def test(model, device, test_loader, scat, display=False):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(scat(data))
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100. * correct / len(test_loader.dataset)
############################################################################
# Train a simple Hybrid Scattering + CNN model on MNIST.

from kymatio import Scattering2D
import torch.optim
import math

for classifier in ['linear','mlp','cnn']:
    # Use second order scattering
    scattering = Scattering2D(M=28, N=28, J=2)
    K = 81 #Number of output coefficients for each spatial postiion

    if use_cuda:
        scattering = scattering.cuda()

    if classifier == 'cnn':
        # Scattering and CNN hybrid
        model = nn.Sequential(
            View(K, 7, 7),
            nn.BatchNorm2d(K),
            nn.Conv2d(K, 64, 3,padding=1), nn.ReLU(),
            View(64*7*7),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, 10)
        ).to(device)

    elif classifier == 'mlp':
        # Evaluate a small fully connected network on scattering
        model = nn.Sequential(
            View(K, 7, 7),
            nn.BatchNorm2d(K),
            View(K*7*7),
            nn.Linear(K*7*7, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 10)
        )

    elif classifier == 'linear':
        # Evaluate linear model on top of scattering
        model = nn.Sequential(
            View(K, 7, 7),
            nn.BatchNorm2d(K),
            View(K * 7 * 7),
            nn.Linear(K * 7 * 7, 10)
        )
    else:
        raise ValueError('Classifier should be cnn/mlp/linear')

    model.to(device)

    ### Initialize the model
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            m.weight.data.normal_(0, 2./math.sqrt(n))
            m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 2./math.sqrt(m.in_features))
            m.bias.data.zero_()

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                                weight_decay=0.0005)
    for epoch in range(1, 16):
        train( model, device, train_loader, optimizer, scattering)

    acc = test(model, device, test_loader, scattering, display = True)
    print('Scattering order  + ' + classifier
          + ' test accuracy: %.2f'%(acc)
          )
