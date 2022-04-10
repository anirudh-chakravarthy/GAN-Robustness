import torch
import torch.nn as nn

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
    def __init__(self, model, loss_fn, eps=0.3, alpha=2/255, steps=40, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']
        self.loss_fn = IDMRFLoss() # loss_fn # nn.MSELoss()

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
            images = self.model(latent, noise=noise, step=step, alpha=alpha, mean_style=mean_style, 
                                style_weight=style_weight, mixing_range=mixing_range)
        images = (images - images.min()) / (images.max() - images.min())
        latent = latent.clone().detach().to(self.device)

        adv_latent = latent.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_latent = adv_latent + torch.empty_like(adv_latent).uniform_(-self.eps, self.eps)
            adv_latent = adv_latent.detach()
            # adv_latent = torch.clamp(adv_latent, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_latent.requires_grad = True
            outputs = self.model(adv_latent, noise=noise, step=step, alpha=alpha, mean_style=mean_style, 
                                 style_weight=style_weight, mixing_range=mixing_range)
            outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min())

            # Calculate loss
            loss = self.loss_fn(outputs, images) # .sum()
            print(loss)

            # Update adversarial images
            grad = torch.autograd.grad(loss, adv_latent,
                                       retain_graph=False, create_graph=False)[0]

            adv_latent = adv_latent.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_latent - latent, min=-self.eps, max=self.eps)
            adv_latent = (latent + delta).detach()
            # adv_latent = torch.clamp(latent + delta, min=0, max=1).detach()

        return adv_latent
