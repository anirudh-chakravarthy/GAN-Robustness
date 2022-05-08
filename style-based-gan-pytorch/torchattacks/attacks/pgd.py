import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attack import Attack
from ..losses import IDMRFLoss


class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, discriminator, eps=0.3, alpha=2/255, steps=40, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']
        self.discriminator = discriminator
        self.loss_fn = nn.MSELoss() # IDMRFLoss()

    def forward(self, 
                latent, 
                noise,
                step=0,
                alpha=-1,
                mean_style=None,
                style_weight=0,
                mixing_range=(-1, -1),):
        r"""
        Overridden.
        """
        with torch.no_grad():
            images = self.model(
                latent, noise=noise, step=step, alpha=alpha, mean_style=mean_style, 
                style_weight=style_weight, mixing_range=mixing_range)
        latent = latent.clone().detach().to(self.device)
        
        adv_latent = latent.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_latent = adv_latent + torch.empty_like(adv_latent).uniform_(-self.eps, self.eps)
            adv_latent = adv_latent.detach()

        adv_latents = []
        for _ in range(self.steps):
            self.model.zero_grad()
            self.discriminator.zero_grad()
            adv_latent.requires_grad = True
            outputs = self.model(
                adv_latent, noise=noise, step=step, alpha=alpha, mean_style=mean_style, 
                style_weight=style_weight, mixing_range=mixing_range)

            # Calculate discriminator loss
            disc_out = self.discriminator(outputs, step=step, alpha=alpha)
            loss = F.softplus(disc_out).mean()
            # print('Disc:', loss.item())

            # drive outputs to zeros, adv attack needs to actually minimize this loss
            # loss = -self.loss_fn(outputs, torch.zeros_like(outputs)) # .sum()

            # enforce KL divergence for adversarial latents
            mu = torch.mean(adv_latent, dim=0)
            sigma = torch.std(adv_latent, dim=0)
            kl_loss = -0.5 * (1. + (sigma **2).log() - mu **2 - sigma **2).mean()
            # print('KL:', kl_loss.item())

            # we perform gradient ascent but should still minimize KL div
            loss = loss - 125. * kl_loss
            # print('Wrong predictions: {} / {}'.format(
            #     (disc_out.sigmoid() < 0.5).sum().item(), disc_out.shape[0]))

            # Calculate loss
            # loss = self.loss_fn(outputs, images) # .sum()

            loss.backward()
            # print('Loss:', loss.item())

            # Update adversarial images
            # grad = torch.autograd.grad(loss, adv_latent, create_graph=False)[0]
            grad = adv_latent.grad
            adv_latent = adv_latent.detach() + self.alpha * grad.sign()
            # delta = torch.clamp(adv_latent - latent, min=-self.eps, max=self.eps)
            # adv_latent = (latent + delta).detach()
            adv_latent = adv_latent.detach()
            adv_latents.append(adv_latent)

        return adv_latents