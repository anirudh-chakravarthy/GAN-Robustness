'''
    Usage: python generate_adversarial.py --size 256 stylegan-256px-new.model
'''
import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torchvision import utils
from sklearn.decomposition import PCA

from model import StyledGenerator, Discriminator
import lpips
from utils.plot_utils import plot_trajectory
from torchattacks import PGD

import pdb


def visualize_data_distribution(z, adv_z, fileName='vis/latent_space_vis.png'):
    z = project_distribution(z)
    adv_z = project_distribution(adv_z)
    plt.scatter(z[:, 0], z[:, 1], label='Orig', c='#6A10F1', alpha=0.80)
    plt.scatter(adv_z[:, 0], adv_z[:, 1], label='Adv', c='#14F110', alpha=0.80)
    plt.legend()
    plt.savefig(fileName)
    plt.close()


def project_distribution(z, dim=2):
    pca = PCA(dim)
    z_proj = pca.fit_transform(z)
    return z_proj


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
    z_list, adv_z_list = [], []
    images, adv_images = [], []
    for i in range(1):
        z = torch.randn(n_sample, 512).to(device)
        noise = []
        for i in range(step + 1):
            size = 4 * 2 ** i
            noise.append(torch.randn(z.shape[0], 1, size, size, device=z.device))

        adversary = PGD(generator, discriminator, alpha=1,
                        steps=100, random_start=True, eps=0.05)
        adv_z = adversary(
            z,
            noise,
            step=step,
            alpha=1,
            mean_style=mean_style,
            style_weight=0.7,
        )

        z_list.extend(z)
        adv_z_list.extend(adv_z[-1])

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
                adv_z[-1],
                noise=noise,
                step=step,
                alpha=1,
                mean_style=mean_style,
                style_weight=0.7,
            )
            images.extend(image)
            adv_images.extend(adv_image)
    
    # plot trajectory of attacked latent vector
    proj_x, proj_y = [], []
    for code in adv_z:
        projected_code = project_distribution(code.detach().cpu().numpy())
        proj_x.append(projected_code[1, 0])
        proj_y.append(projected_code[1, 1])
    plot_trajectory(proj_x, proj_y, 'vis/latent_code_vis.mp4')

    z = torch.stack(z_list, dim=0)
    adv_z = torch.stack(adv_z_list, dim=0)

    # plot PCA
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
    

def attack_w_space(
    generator, 
    discriminator, 
    step, 
    mean_style, 
    n_sample, 
    device
):
    
    z_list = []
    images, attacked_images = [], []
    for i in range(1):
        z = torch.randn(n_sample, 512).to(device)
        noise = []
        for i in range(step + 1):
            size = 4 * 2 ** i
            noise.append(torch.randn(z.shape[0], 1, size, size, device=z.device))

        z_list.extend(z)
        
        image, attacked_image = generator(
            z,
            noise=noise,
            step=step,
            alpha=1,
            mean_style=mean_style,
            style_weight=0.7,
            discriminator=discriminator, 
            attack_w=True
        )
        images.extend(image)
        attacked_images.extend(attacked_image)

    images_tensor = torch.stack(images, dim=0)
    attacked_images_tensor = torch.stack(attacked_images, dim=0)
    return images_tensor, attacked_images_tensor
    

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
    parser.add_argument('disc_path', type=str, help='path to discriminator checkpoint file')

    args = parser.parse_args()
    
    device = 'cuda'
    seed_everything(args.seed)

    ckpt = torch.load(args.path)
    generator = StyledGenerator(512).to(device)
    generator.load_state_dict(ckpt['g_running'])
    generator.eval()

    disc_ckpt = torch.load(args.disc_path) # ckpt
    discriminator = Discriminator(from_rgb_activate=not args.no_from_rgb_activate).to(device)
    discriminator.load_state_dict(disc_ckpt['discriminator'])
    discriminator.eval()

    mean_style = get_mean_style(generator, device)

    step = int(math.log(args.size, 2)) - 2

    # img = sample(generator, step, mean_style, args.n_row * args.n_col, device)
    
    attack_w = True
    if attack_w:
        img, adv_img = attack_w_space(generator, discriminator, step, mean_style, args.n_row * args.n_col, device)
    else:
        img, adv_img = attack(generator, discriminator, step, mean_style, args.n_row * args.n_col, device)
    utils.save_image(img, 'ATTACK_Z_sample.png', nrow=args.n_col, normalize=True, range=(-1, 1))
    utils.save_image(adv_img, 'ATTACK_Z_sample_adv.png', nrow=args.n_col, normalize=True, range=(-1, 1))
    
#     for j in range(20):
#         img = style_mixing(generator, step, mean_style, args.n_col, args.n_row, device)
#         utils.save_image(
#             img, f'sample_mixing_{j}.png', nrow=args.n_col + 1, normalize=True, range=(-1, 1)
#         )
