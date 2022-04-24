import os
import torch
import torch.nn as nn

def get_device():
    """
    :return: device
    """
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

def save_checkpoint(model, optimizer, save_dir, epoch):
    """
    Save model checkpoints

    :param model: model
    :param optimizer: optimizer
    :param save_dir: directory to store model checkpoints
    :param epoch: epoch number
    """
    filename = f'checkpoint_{epoch}.pth'
    filepath = save_dir + filename
    print(".....saving checkpoint")
    checkpoint = {
        "step": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(checkpoint, filepath)


def load_checkpoint( model, optimizer, checkpoint_dir):
    """
    Load saved checkpoint

    :param optimizer: optimizer
    :param model: model
    :param checkpoint_dir: path to saved checkpoint
    :return: iteration where the model should resume training
    """
    start_iter = None
    checkpoint = torch.load(checkpoint_dir, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_iter = checkpoint['step']
    return start_iter

def get_optimizers_and_schedulers(model):
    """
    :param model: input model
    :return: optimizer and scheduler for training
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

    return (
        optimizer,
        scheduler
    )

# ------- UNIT TEST ----------- #


if __name__ == "__main__":
    pass


    