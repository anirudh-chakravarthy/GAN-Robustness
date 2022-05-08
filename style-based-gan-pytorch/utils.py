"""
Common Utilities for Analysis and Plotting.
"""

import torch
from cleanfid import fid
import numpy as np
import matplotlib.pyplot as plt


@torch.no_grad()
def compute_fid(
    latent_vec_1,
    latent_vec_2
):
    """Compute the FID Score between two latent vectors. 

    Keyword Arguments:
    latent_vec_1 -- First Latent Vector for Comparison
    latent_vec_2 -- Second Latent Vector for Comparison

    Returns:
    score -- Computed FID score between the two latent vectors.
    """

    # Get Features for First Latent Vector
    mu_1 = torch.mean(latent_vec_1, axis=0)
    sigma_1 = np.cov(latent_vec_1.cpu().numpy(), rowvar=False)

    # Get Features for the Second Latent Vector
    mu_2 = torch.mean(latent_vec_2, axis=0)
    sigma_2 = np.cov(latent_vec_2.cpu().numpy(), rowvar=False)

    # Compute FID Score
    score = fid.frechet_distance(
        mu_1.cpu().numpy(),
        sigma_1,
        mu_2.cpu().numpy(),
        sigma_2
    )

    return score


@torch.no_grad()
def compute_fid(
    images,
    res=256
):
    """Compute the FID Score between the images and the original dataset.

    Keyword Arguments:
    images -- Input Images

    Returns:
    score -- Computed FID score between dataset and images.
    """

    # Get Features for Dataset: ffhq_clean_trainval_256.npz
    mu_dataset, sigma_dataset = fid.get_reference_statistics(
        name='ffhq',
        res=res,
        mode="clean",
        split="trainval",
        metric="FID"
    )

    # Get Features for the Second Latent Vector
    model = fid.build_feature_extractor(
        mode='clean'
    )

    feat = fid.get_batch_features(
        images,
        model,
        device=torch.device("cuda")
    )

    mu_images = np.mean(
        feat,
        axis=0
    )
    sigma_images = np.cov(
        feat,
        rowvar=False
    )

    # Compute FID Score
    score = fid.frechet_distance(
        mu_dataset,
        sigma_dataset,
        mu_images,
        sigma_images
    )

    return score


def save_plot(
    x,
    y,
    save_path,
    c='green',
):
    """Generic Matplotlib plotting and saving function.

    Keyword Arguments:
    x -- Variable on the x axis.
    y -- Variable on the y axis.
    """

    # Assert the (key, value) pairs are consistently numbered.
    assert(len(x) == len(y))

    # Plot according to given color.
    plt.plot(x, y, c=c)
    plt.savefig(save_path)
