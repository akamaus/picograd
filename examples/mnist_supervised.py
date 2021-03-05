import torch
from torch import nn
import torch.nn.functional as F

from torchvision.datasets import MNIST
from torchvision import transforms

from picograd.configs.train_config import TrainConfig
from picograd.trainers.base import BaseTrainer, BaseContext


class Lenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class MnistContext(BaseContext):
    def compute_loss(self, inp):
        out = self.model(inp[0])
        log_probs = nn.functional.log_softmax(out, dim=1)
        loss = nn.functional.nll_loss(log_probs, inp[1])

        acc = (out.argmax(dim=1) == inp[1]).float().mean()

        self.log_comp.log_metric('loss', loss)
        self.log_comp.log_metric('acc', acc)

        return loss


    
class MnistTrainer(BaseTrainer):
    Context = MnistContext


class MnistConfig(TrainConfig):
    def __init__(self):
        super().__init__()
        self.val_period = 10
        self.learning_rate = 1e-3
        self.use_meta_info = False

    def build_model(self):
        model = Lenet()
        return model

    def build_trainer(self, model, storage):
        train_ds = MNIST(root='mnist_data', download=True, train=True, transform = transforms.ToTensor())
        test_ds = MNIST(root='mnist_data', download=True, train=True, transform = transforms.ToTensor())

        trainer = MnistTrainer(model=model,
                               datasets={'training': train_ds,
                                         'test': test_ds,
                                        },
                               storage=storage, cfg=self)
        return trainer


Config = MnistConfig
