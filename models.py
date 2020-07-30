from torch import nn
from torchvision.models import resnet50,vgg16,vgg19,googlenet,resnet152

ngf = 128
nz = latent_dim = 10
e_lim = 10
nc = 3  # Number of Channels


# Fixed Architecture: Weights will be updated by Backprop.
class AdveraryGenerator(nn.Module):
    def __init__(self, e_lim):
        super(AdveraryGenerator, self).__init__()
        self.e_lim = e_lim
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nz, out_channels=1024, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(256, 128, 4, 2, 2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(128, 64, 4, 2, 2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. (nc) x 64 x 64
            nn.ConvTranspose2d(64, 3, 4, 4, 4, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            nn.Tanh()
        )

    def forward(self, x):
        return self.e_lim * self.main(x)  # Scaling of ε


adversarygen = AdveraryGenerator(e_lim).to(device)