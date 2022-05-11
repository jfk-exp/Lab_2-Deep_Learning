import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import losses.losses
from custom_dataset.mnist_dataset import MNISTDataset
from losses.losses import mse
from models.models import LinearRegression, Classifier
from options.classification_options import ClassificationOptions
from options.linear_regression_options import LinearRegressionOptions
from options.options import Options


def plot_tensor(to_plot: torch.Tensor, title: str):
    gray_image_tensor = to_plot.view([1, -1, 1])
    numpy_im = gray_image_tensor.numpy()
    plt.imshow(numpy_im, cmap=plt.get_cmap("GnBu"), interpolation="none", vmin=0,
               vmax=1)
    plt.title(title)
    plt.show()


def plot_rgb_tensor(to_plot: torch.Tensor, title: str):
    fig = plt.figure()
    plt.title(title)
    plt.imshow(transforms.ToPILImage()(to_plot), interpolation="None")
    fig.show()


def train_lin_model(model: LinearRegression, optimizer: torch.optim.Optimizer,
                    train_data: DataLoader, options: LinearRegressionOptions):
    for epoch in range(options.num_epochs):
        for data in train_data:
            size, price = data[:, 0].unsqueeze(1).to(options.device), data[:, 1].unsqueeze(1).to(options.device)
            """START TODO: implement some missing parts. look at the comments to see what needs to be done."""
            # Forward the size data through the model
            output = model(size)

            # calculate the loss, use your self created mse loss
            loss=mse(output, price)

            # As mentioned before, the grads always needs to be zeroed before backprop (use your optimizer to do this)
            optimizer.zero_grad()
            # propagate the loss backward
            loss.backward()
            # use your optimizer to perform an update step
            optimizer.step()
            """END TODO"""
        print(f'epoch [{epoch + 1}/{options.num_epochs}]: ', end="")
        test_lin_reg_model(model, train_data)


def test_lin_reg_model(model: LinearRegression, test_data: DataLoader):
    with torch.no_grad():
        loss = 0
        for data in test_data:
            size, price = data[:, 0].unsqueeze(1), data[:, 1].unsqueeze(1)
            estimated_price = model(size)
            loss = loss + torch.sqrt(mse(estimated_price, price))
        print(f'Avg error/example: € {loss / len(test_data.dataset) :.2f}\n')


def print_lin_reg(model: LinearRegression, options: LinearRegressionOptions):
    print(f"Actual function: f(x) = 5000 * x + 100 000 + {options.noise_house_data} * N(0, 1).")
    print(
        f"Estimated by linear regression: h(x) = {model.linear_layer.weight.data[0].item()} * x + {model.linear_layer.bias.data[0].item()}")


def test_lin_reg_plot(model: LinearRegression, test_data: DataLoader, options: LinearRegressionOptions):
    """
    Show some examples of the selected dataset.
    """
    fig = plt.figure()

    # plot real and estimated data points
    with torch.no_grad():
        for data in test_data:
            plt.scatter(data[:, 0], data[:, 1], c="g")
            size, price = data[:, 0].unsqueeze(1), data[:, 1].unsqueeze(1)
            estimated_price = model(size)
            plt.scatter(data[:, 0], estimated_price, c="r")

        # plot line
        x = torch.linspace(options.min_house_size, options.max_house_size, 50000, device=options.device)
        plt.plot(x.numpy(), 5000 * x + 100000, "g")
        plt.plot(x.numpy(), model(x.unsqueeze(1)).numpy(), "r")

    plt.title("Data")
    plt.xlabel("size [m^2]")
    plt.ylabel("Price [€]")
    plt.legend(["Unknown function f(x)", "Linear regression line", "Real data samples", "Estimated data samples"])
    plt.plot()

    fig.show()


def train_classification_model(model: Classifier, optimizer: torch.optim.Optimizer,
                               dataset: MNISTDataset, options: ClassificationOptions):
    """START TODO: select an appropriate criterion (loss function)"""
    criterion = torch.nn.CrossEntropyLoss()

    """END TODO"""
    for epoch in range(options.num_epochs):
        running_loss = 0
        for x, y in dataset.train_loader:
            """START TODO: fill in the gaps as mentioned by the comments"""
            # forward the data x through the model.
            # Note: x does not have the correct shape,
            # it should become (batch_size, -1), where the size -1 is inferred from other dimensions
            # (see TORCH.TENSOR.VIEW on the PyTorch documentation site)
            x = torch.reshape(x, (x.shape[0], -1))
            #if(x.size() != (64,784)):
               #print("error")
                # continue
            result = model(x)
            result.float()


            # calculate the loss, use your previously defined criterion
            loss = criterion(result, y)
            # zero out all gradients
            optimizer.zero_grad()

            # propagate the loss backward
            loss.backward()
            # use your optimizer to perform an update step
            optimizer.step()
            """END TODO"""
            running_loss += loss.item()
        print(f'epoch [{epoch + 1}/{options.num_epochs}]: ', end="")
        print(f"Running loss = {running_loss / len(dataset.train_loader)}")
        test_classification_model(model, dataset, options)


def test_classification_model(model: Classifier, dataset: MNISTDataset, options: ClassificationOptions):
    with torch.no_grad():
        tot = 0
        correct = 0

        for x, y in dataset.test_loader:
            output = model(x.view(x.shape[0], -1).to(options.device))

            # choose the number with the highest probability as prediction
            _, predicted = torch.max(output, dim=1)
            tot += y.size(0)
            correct += (predicted == y).sum().item()
        print(f'Accuracy: {100 * correct / tot :.2f}%')


def classify_images(model: Classifier, dataset: MNISTDataset, options: ClassificationOptions):
    with torch.no_grad():
        examples = enumerate(dataset.test_loader)
        _, (x, y) = next(examples)

        output = model(x.view(x.shape[0], -1).to(options.device))

        # choose the number with the highest probability as prediction
        predicted = torch.argmax(output, dim=1)

        fig = plt.figure()
        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()
            plt.imshow(x[i][0], cmap='gray', interpolation='none')
            plt.title(f"y: {y[i]}, estimation: {predicted[i].item()}")
            plt.xticks([])
            plt.yticks([])
        fig.show()


def not_implemented() -> str:
    return "NOT IMPLEMENTED"


def save(model: nn.Module, options: Options):
    if not os.path.exists(options.save_path):
        os.makedirs(options.save_path)
    torch.save(model.state_dict(), options.save_path + options.model_name)


def load(model: nn.Module, options: Options):
    try:
        model.load_state_dict(torch.load(options.load_path + options.model_name))
        model.eval()
    except IOError:
        print("Could not load module!!")


def init_pytorch(options: Options):
    # set all random seeds for reproducibility
    torch.manual_seed(options.random_seed)
    torch.cuda.manual_seed(options.random_seed)
    random.seed(options.random_seed)
    # set device
    if options.device == "cuda" and torch.cuda.is_available():
        options.device = torch.device("cuda:0")
    else:
        options.device = torch.device("cpu")
