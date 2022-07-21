import os
import torch
import tarfile
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.utils import download_url

matplotlib.rcParams['figure.facecolor'] = '#ffffff'

# Downloading the dataset
# dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
# download_url(dataset_url, '.')

# Extracting from archive
# with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
#    tar.extractall(path='./data')

# Directory for the dataset
data_dir = './data/cifar10'

# Creating the dataset
dataset = ImageFolder(data_dir+'/train', transform=ToTensor())

# Checking the shape
# img, label = dataset[0]
# print(img.shape, label)
# print(img)


# Function for displaying an image and its label
def show_example(img, label):
    print('Label: ', dataset.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1, 2, 0))
    plt.show()


# show_example(*dataset[0])


# Function for splitting the dataset
def split_indices(n, val_pct, seed):
    # Determining size of validation set
    n_val = int(val_pct*n)
    # Setting the random seed (for reproducibility)
    np.random.seed(seed)
    # Creating random permutation of 0 to n-1
    idxs = np.random.permutation(n)
    # Picking first n_val indices for validation set
    return idxs[n_val:], idxs[:n_val]


# Setting the percentage for validation set and seed
val_pct = 0.2
rand_seed = 42

# Splitting the dataset
train_indices, val_indices = split_indices(len(dataset), val_pct, rand_seed)

# Checking the split dataset
# print(len(train_indices), len(val_indices))
# print('Sample validation indices: ', val_indices[:10])

# Batch size
batch_size = 100

# Training sampler and data loader
train_sampler = SubsetRandomSampler(train_indices)
train_dl = DataLoader(dataset,
                      batch_size,
                      sampler=train_sampler)

# Validation sampler and data loader
val_sampler = SubsetRandomSampler(val_indices)
val_dl = DataLoader(dataset,
                    batch_size,
                    sampler=val_sampler)

# Example model
# simple_model = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1), nn.MaxPool2d(2, 2))

# Checking the difference between the shape of input and output
# for images, labels in train_dl:
#    print('images.shape: ', images.shape)
#    out = simple_model(images)
#    print('out.shape: ', out.shape)
#    break


# Model definition with helper methods
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        # Generate predictions
        out = self(images)
        # Calculate loss
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        # Generate predictions
        out = self(images)
        # Calculate loss
        loss = F.cross_entropy(out, labels)
        # Calculate accuracy
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        # Combine losses
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        # Combine accuracies
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


# Function that calculates the accuracy of the model
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# Base model
class Cifar10CnnModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))

    def forward(self, xb):
        return self.network(xb)


# Calling the model
model = Cifar10CnnModel()
# Checking the shape of images after using the model
# for images, labels in train_dl:
#    print('images.shape: ', images.shape)
#    out = model(images)
#    print('out.shape: ', out.shape)
#    print('out[0]: ', out[0])
#    break


# Using the GPU
def get_default_device():
    """Picking GPU if it's available"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Moving tensor(s) to the chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """Wrapping a dataloader to move data to device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# Loading data onto GPU
device = get_default_device()
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
to_device(model, device)


# Evaluate calculates the overall loss
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


# Actual training loop
def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# Instantiating the model
model = to_device(Cifar10CnnModel(), device)

# Model accuracy before training
# print(evaluate(model, val_dl))

# Hyperparameters
num_epochs = 10
opt_func = torch.optim.Adam
lr = 0.001

# Training
# history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)


# Graph for accuracy over epochs
def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
# plot_accuracies(history)
# plt.show()


# Saving the trained model
# torch.save(model.state_dict(), 'cifar10-logistics.pth')

# Loading already trained model
model2 = to_device(Cifar10CnnModel(), device)
model2.load_state_dict(torch.load('cifar10-logistics.pth'))


# Function for testing the model
def predict_image(img, model):
    # Converting to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Picking an index with highest probability
    _, preds = torch.max(yb, dim=1)
    # Retrieving the class label
    return dataset.classes[preds[0].item()]


# Testing the model
test_dataset = ImageFolder(data_dir+'/test', transform=ToTensor())
img, label = test_dataset[100]
plt.imshow(img.permute(1, 2, 0))
print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model2))
plt.show()
