
import torch
import torch.nn as nn
import sys 
sys.path.append("..")
from utils.utils import get_device

def mse_loss(pred, target, *args, **kwargs):
    """ 
    MSE loss between pred and target tensors

    Args:
        pred: predicted image feature (batch_size, 512)
        target: target image feature (batch_size, 512)
    Returns:
        mse loss betweent 
    """

    criterion = nn.MSELoss(reduction='sum')
    loss = criterion(pred, target) / pred.shape[0]
    return loss

def gradient_penalty(real, fake, disc, epsilon, device='cuda'):
    """
    Computes gradient penalty term for WGAN-GP
    
    Args:
        real:
        fake:
        disc:
        epsilon:

    Returns:
    """

    interp = real * epsilon + fake * (1 - epsilon)
    interp.requires_grad = True
    disc_interp = disc(interp)
    gradient = torch.autograd.grad(
        inputs=interp,
        outputs=disc_interp,
        grad_outputs=torch.ones_like(disc_interp),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gp = torch.mean((gradient_norm - 1) ** 2)
    return gp

def discriminator_loss(real, fake, disc, discrim_real, discrim_fake, lamb, device='cuda'):
    """
    WGAN-GP loss for discriminator.
    loss = max_D E[D(real_data)] - E[D(fake_data)] + lambda * E[(|| grad wrt interpolated_data (D(interpolated_data))|| - 1)^2]
    
    Args:
        real:
        fake:
        disc:
        discrim_real:
        discrim_fake:
        lamb:

    Returns:

    """
    if not device:
        device = get_device()
    else:
        device = device

    B, C = real.shape
    epsilon = torch.rand(size=(B, 1)).repeat(1, C).to(device)
    gp = gradient_penalty(real, fake, disc, epsilon)
    loss_discrim = -1 * (torch.mean(discrim_real) - torch.mean(discrim_fake)) + lamb * gp
    return loss_discrim


def generator_loss(real, fake, discrim_fake, device='cuda'):
    """
    loss = - E[D(fake_data)]
    output = disc(discrim_fake)
    """
    loss_gen = -1 * torch.mean(discrim_fake) + mse_loss(fake, real)
    return loss_gen


class Losses(nn.Module):
    """ 
    Class to create a loss object
    """
    def __init__(self, fn='adv_disc_loss', device=None):
        """
        Args: 
            fn: Defines the loss function to be used
        """
        super(Losses, self).__init__()

        self.fn = loss_dict[fn]
        if not device:
            self.device = get_device()
        else:
            self.device = device

    def forward(self, *args, **kwargs):
        return self.fn(device=self.device, *args, **kwargs)        


loss_dict = {
    'adv_gen_loss': generator_loss,
    'adv_disc_loss': discriminator_loss
    }


# ------- UNIT TEST ----------- #

def test_adv_loss():

    device = get_device()
    real = torch.randn(size=(10, 512)).to(device)
    fake = torch.randn(size=(10, 512)).to(device)
    disc = nn.Linear(512, 1).to(device)
    gen = nn.Linear(512, 512).to(device)
    disc.train()
    gen.train()
    lamb = 10
    loss_disc = Losses('adv_disc_loss')
    loss_gen = Losses('adv_gen_loss')

    print(loss_disc(real, fake, disc, disc(real), disc(fake), lamb))
    print(loss_gen(real, fake, disc(fake)))

if __name__ == "__main__":
    test_adv_loss()