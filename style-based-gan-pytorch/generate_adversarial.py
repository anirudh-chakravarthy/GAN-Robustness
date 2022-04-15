'''
    Usage: python generate_adversarial.py --size 256 stylegan-256px-new.model
'''
import argparse
import math
import numpy as np
import random

import torch
from torchvision import utils

from model import StyledGenerator, Discriminator
import lpips

from torchattacks import PGD
import pdb

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def visualize_data_distribution(z, adv_z):
    pca = PCA(2)

    z_proj = pca.fit_transform(z)
    adv_z_proj = pca.fit_transform(adv_z)

    plt.scatter(z_proj[:, 0], z_proj[:, 1], label='Orig', c='#6A10F1', alpha=0.80)
    plt.scatter(adv_z_proj[:, 0], adv_z_proj[:, 1], label='Adv', c='#14F110', alpha=0.80)
    plt.legend()
    plt.savefig(f'latent_space_vis.png')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def get_mean_style(generator, device):
    mean_style = None

    for i in range(10):
        style = generator.mean_style(torch.randn(1024, 512).to(device))

        if mean_style is None:
            mean_style = style

        else:
            mean_style += style

    mean_style /= 10
    return mean_style

@torch.no_grad()
def sample(generator, step, mean_style, n_sample, device):
    image = generator(
        torch.randn(n_sample, 512).to(device),
        step=step,
        alpha=1,
        mean_style=mean_style,
        style_weight=0.7,
    )
    
    return image


def attack(generator, discriminator, step, mean_style, n_sample, device):
#     loss_fn = lpips.PerceptualLoss(
#         model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
#     )
    
    z_list, adv_z_list = [], []
    images, adv_images = [], []
    for i in range(1):
        z = torch.randn(n_sample, 512).to(device)
        noise = []
        for i in range(step + 1):
            size = 4 * 2 ** i
            noise.append(torch.randn(z.shape[0], 1, size, size, device=z.device))

        adversary = PGD(generator, discriminator, alpha=1,
                        steps=100, random_start=False, eps=0.05)
        adv_z = adversary(
            z,
            noise,
            step=step,
            alpha=1,
            mean_style=mean_style,
            style_weight=0.7,
        )

        z_list.extend(z)
        adv_z_list.extend(adv_z)

#         visualize_data_distribution(
#             z.detach().cpu().numpy(), adv_z.detach().cpu().numpy())

#         mu = torch.mean(z, dim=0)
#         sigma = torch.std(z, dim=0)
#         adv_mu = torch.mean(adv_z, dim=0)
#         adv_sigma = torch.std(adv_z, dim=0)
#         print(-0.5 * (1. + (sigma **2).log() - mu **2 - sigma **2).mean())
#         print(-0.5 * (1. + (adv_sigma **2).log() - adv_mu **2 - adv_sigma **2).mean())
#         print(torch.mean((z-adv_z) **2).item())

        with torch.no_grad():
            image = generator(
                z,
                noise=noise,
                step=step,
                alpha=1,
                mean_style=mean_style,
                style_weight=0.7,
            )
            adv_image = generator(
                adv_z,
                noise=noise,
                step=step,
                alpha=1,
                mean_style=mean_style,
                style_weight=0.7,
            )
            images.extend(image)
            adv_images.extend(adv_image)
    
    z = torch.stack(z_list, dim=0)
    adv_z = torch.stack(adv_z_list, dim=0)
    visualize_data_distribution(
        z.detach().cpu().numpy(), adv_z.detach().cpu().numpy())

    mu = torch.mean(z, dim=0)
    sigma = torch.std(z, dim=0)
    adv_mu = torch.mean(adv_z, dim=0)
    adv_sigma = torch.std(adv_z, dim=0)
    print('Sampled KL:', -0.5 * (1. + (sigma **2).log() - mu **2 - sigma **2).mean().item())
    print('Adversarial KL:', -0.5 * (1. + (adv_sigma **2).log() - adv_mu **2 - adv_sigma **2).mean().item())

    image = torch.stack(images, dim=0)
    adv_image = torch.stack(adv_images, dim=0)
    return image, adv_image
    

@torch.no_grad()
def style_mixing(generator, step, mean_style, n_source, n_target, device):
    source_code = torch.randn(n_source, 512).to(device)
    target_code = torch.randn(n_target, 512).to(device)
    
    shape = 4 * 2 ** step
    alpha = 1

    images = [torch.ones(1, 3, shape, shape).to(device) * -1]

    source_image = generator(
        source_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    )
    target_image = generator(
        target_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    )

    images.append(source_image)

    for i in range(n_target):
        image = generator(
            [target_code[i].unsqueeze(0).repeat(n_source, 1), source_code],
            step=step,
            alpha=alpha,
            mean_style=mean_style,
            style_weight=0.7,
            mixing_range=(0, 1),
        )
        images.append(target_image[i].unsqueeze(0))
        images.append(image)

    images = torch.cat(images, 0)
    
    return images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=1024, help='size of the image')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--n_row', type=int, default=3, help='number of rows of sample matrix')
    parser.add_argument('--n_col', type=int, default=5, help='number of columns of sample matrix')
    parser.add_argument(
        '--no_from_rgb_activate',
        action='store_true',
        help='use activate in from_rgb (original implementation)',
    )
    parser.add_argument('path', type=str, help='path to checkpoint file')
#     parser.add_argument('disc_path', type=str, help='path to discriminator checkpoint file')

    args = parser.parse_args()
    
    device = 'cuda'
    seed_everything(args.seed)

    ckpt = torch.load(args.path)
    generator = StyledGenerator(512).to(device)
    generator.load_state_dict(ckpt['g_running'])
    generator.eval()

    disc_ckpt = ckpt # torch.load(args.path)
    discriminator = Discriminator(from_rgb_activate=not args.no_from_rgb_activate).to(device)
    discriminator.load_state_dict(disc_ckpt['discriminator'])
    discriminator.eval()

    mean_style = get_mean_style(generator, device)

    step = int(math.log(args.size, 2)) - 2

    # img = sample(generator, step, mean_style, args.n_row * args.n_col, device)
    img, adv_img = attack(generator, discriminator, step, mean_style, args.n_row * args.n_col, device)
    utils.save_image(img, 'sample.png', nrow=args.n_col, normalize=True, range=(-1, 1))
    utils.save_image(adv_img, 'sample_adv.png', nrow=args.n_col, normalize=True, range=(-1, 1))
    
#     for j in range(20):
#         img = style_mixing(generator, step, mean_style, args.n_col, args.n_row, device)
#         utils.save_image(
#             img, f'sample_mixing_{j}.png', nrow=args.n_col + 1, normalize=True, range=(-1, 1)
#         )
