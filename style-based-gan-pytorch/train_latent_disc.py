'''
    Usage: python train_latent_disc.py --ckpt stylegan-256px-new.model
'''
import argparse
from copy import deepcopy

import torch
from torch import nn, optim
from torch.autograd import grad
from torch.nn import functional as F
from tqdm import tqdm

from losses.losses import Losses
from model import StyledGenerator, Discriminator
from models import LatentDiscriminator, AdversarialMapper

import pdb


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def train(
    args, 
    generator,
    adv_generator,
    noise_discriminator,
    latent_discriminator,
):
    requires_grad(noise_discriminator, True)
    requires_grad(latent_discriminator, True)
    requires_grad(generator, True)
    requires_grad(adv_generator, True)

    latent_disc_loss_fn = Losses(fn=args.loss)
    latent_disc_loss_val = 0.

    step = 6
    alpha = 1

    pbar = tqdm(range(12_000))
    for i in pbar:
        generator.zero_grad()
        adv_generator.zero_grad()
        noise_discriminator.zero_grad()
        latent_discriminator.zero_grad()

        d_optimizer.zero_grad()
        m_optimizer.zero_grad()

        z = torch.rand(args.batch_size, 512).cuda()
        real_w = generator.module.style(z)
        fake_w = adv_generator.module.style(z) # real_w
        
        # TODO: call generator on real and fake W
        # gen_real = generator(z) # update args
        gen_fake = adv_generator(z, step=step, alpha=alpha) # update args
        
        # compute noise discriminator loss on real image
        # gen_scores = noise_discriminator(gen_real)
        # gen_predict = F.softplus(-gen_scores).mean()
        # gen_predict.backward(retain_graph=True)

        # grad_real = grad(
        #     outputs=gen_scores.sum(), inputs=gen_real, create_graph=True
        # )[0]
        # grad_penalty = (
        #     grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
        # ).mean()
        # grad_penalty = 10 / 2 * grad_penalty
        # grad_penalty.backward()
        # grad_loss_val = grad_penalty.item()
        
        # compute noise discriminator loss on fake image
        fake_predict = noise_discriminator(gen_fake, step=step, alpha=alpha)
        fake_predict = F.softplus(fake_predict).mean()
        fake_predict.backward()
        
        latent_disc_loss = latent_disc_loss_fn(real_w, fake_w, latent_discriminator, 5.0)
        latent_disc_loss_val = latent_disc_loss.item()
        latent_disc_loss.backward()

        d_optimizer.step()
        m_optimizer.step()

        if i % 500 == 0:
            torch.save({
                    'adv_mapper': adv_generator.module.style.state_dict(),
                    'discriminator': latent_discriminator.state_dict(),
                    'm_optimizer': m_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                },
                f'checkpoint_latent_discriminator/train_step-{i}.model',
            )

        state_msg = (f'D: {latent_disc_loss_val:.3f}')
        pbar.set_description(state_msg)


if __name__ == '__main__':
    code_size = 512
    parser = argparse.ArgumentParser(description='Training a latent discriminator')

    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument(
        '--ckpt', default=None, type=str, help='load from previous checkpoints'
    )
    parser.add_argument(
        '--no_from_rgb_activate',
        action='store_true',
        help='use activate in from_rgb (original implementation)',
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='adv_disc_loss',
        choices=['adv_disc_loss', 'adv_gen_loss'],
        help='class of gan loss',
    )

    args = parser.parse_args()
    generator = nn.DataParallel(StyledGenerator(code_size)).cuda()
    adv_generator = nn.DataParallel(StyledGenerator(code_size)).cuda()
    noise_discriminator = nn.DataParallel(
        Discriminator(from_rgb_activate=not args.no_from_rgb_activate)).cuda()
    latent_discriminator = nn.DataParallel(LatentDiscriminator()).cuda()

     # TODO: Change generator to ProGAN generator
    # latent_mapper = nn.DataParallel(generator.module.style)
    # adv_mapper = deepcopy(latent_mapper)

    d_optimizer = optim.Adam(latent_discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))
    m_optimizer = optim.Adam(
        adv_generator.module.style.parameters(), lr=args.lr, betas=(0.0, 0.99))

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        generator.module.load_state_dict(ckpt['generator'])
        adv_generator.module.load_state_dict(ckpt['generator'])

    args.batch_size = 16

    train(args, generator, adv_generator, noise_discriminator, latent_discriminator)