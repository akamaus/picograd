import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import MNIST
from torchvision import transforms
import torchvision

from picograd.configs.train_config import TrainConfig
from picograd.trainers.base import BaseTrainer, BaseContext


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super().__init__()
        self.meta_parameters = {'args': {'nz':nz, 'ngf':ngf, 'nc':nc} }

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     nc, 4, 2, 1, bias=False),
            nn.Sigmoid(),
            # state size. (ngf) x 32 x 32
            #nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            # state size. (nc) x 64 x 64
        )

        self.apply(weights_init)

    def forward(self, input):
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.meta_parameters = {'args': {'nc': nc, 'ndf': ndf}}

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 2, bias=False),
        )

        self.apply(weights_init)

    def forward(self, input):
        output = self.main(input)

        return output.view(-1, 1).squeeze(1)


class GanContext(BaseContext):
    REAL = 1
    FAKE = 0
    def compute_loss(self, batch):
        gen = self.model['gen']
        disc = self.model['disc']

        # Train D on real
        inp_real = batch[0]
        batch_size = inp_real.size(0)
        label = torch.full((batch_size,), self.REAL,
                           dtype=inp_real.dtype, device=inp_real.device)

        output = self.model['disc'](inp_real)
        loss_D_real = F.binary_cross_entropy_with_logits(output, label)

        disc.zero_grad()
        loss_D_real.backward()

        probs = F.sigmoid(output)
        acc_D_real = (probs > 0.5).float().mean()

        # Train D on fake
        label.fill_(self.FAKE)
        noise = torch.randn(inp_real.size(0), self.trainer.cfg.nz,1, 1, device=inp_real.device)
        inp_fake = gen(noise)
        output = disc(inp_fake.detach())

        loss_D_fake = F.binary_cross_entropy_with_logits(output, label)
        loss_D_fake.backward()

        probs = F.sigmoid(output)
        acc_D_fake = (probs < 0.5).float().mean()

        self.optimizer['disc'].step()

        # Train G
        label.fill_(self.REAL)  # try to trick discriminator
        output = disc(inp_fake)
        loss_G = F.binary_cross_entropy_with_logits(output, label)

        gen.zero_grad()
        loss_G.backward()

        self.optimizer['gen'].step()

        self.log_comp.log_metric('loss_D_real', loss_D_real)
        self.log_comp.log_metric('loss_D_fake', loss_D_fake)
        self.log_comp.log_metric('loss_G', loss_G)
        self.log_comp.log_metric('loss_D', loss_D_real + loss_D_fake)
        self.log_comp.log_metric('acc_D_real', acc_D_real)
        self.log_comp.log_metric('acc_D_fake', acc_D_fake)
        self.log_comp.log_metric('acc_D', (acc_D_real+ acc_D_fake)* 0.5)


class GanTrainer(BaseTrainer):
    Context = GanContext

    def update_model(self, ctx, loss):
        # update steps are merged with loss computation
        pass

    def validation(self):
        gen = self.model['gen']

        if not hasattr(self, 'noise'):
            self.noise = torch.randn(25, self.cfg.nz, 1, 1, device=gen.device)

        with torch.no_grad():
            fake = gen(self.noise)

        image_plate = torchvision.utils.make_grid(fake, nrow=5)
        self.contexts['training'].log_comp.log_image('fake_sample', image_plate)


class MnistGanConfig(TrainConfig):
    def __init__(self):
        super().__init__()
        self.learning_rate = [2e-4, 1e-5]
        self.nz = 100
        self.ngf = 64
        self.ndf = 64

        self.batch_size = 128
        self.epoch_size = None
        self.val_period = 1

        self.use_meta_info = False
        self.allow_nulls = True
        self.device = 'cuda'

    def build_model(self):
        gen = Generator(self.nz, self.ngf, 1)
        disc = Discriminator(1, self.ndf)
        return {'gen': gen, 'disc': disc}

    def build_trainer(self, model, storage):
        train_ds = MNIST(root='mnist_data', download=True, train=True,
                         transform=transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()]))

        trainer = GanTrainer(model=model,
                             datasets={'training': train_ds
                                      },
                               storage=storage, cfg=self)
        return trainer


Config = MnistGanConfig
