from typing import Collection
import matplotlib.pyplot as plt
import os

class EmptyLoader():
    '''
    An empty dataloader that always returns None when iterated through.
    To be used by zipping two DataLoaders together in place of the one which is None.
    '''
    def __iter__(self):
        return self
    def __next__(self):
        return None

def loss_plotter(loss_g:Collection, loss_d:Collection, figpath:str) -> None:
    '''
    Plots the losses for the generator and the discriminator after a GAN train.

    Parameters:
    ------------------
    loss_g: a collection of floats, each representing a loss value for the generator as the train is performed
    loss_d: a collection of floats, each representing a loss value for the discriminator as the train is performed
    figname: a string containing the path where the image is saved. '.png' is always appended to said path.
    '''
    os.makedirs(os.path.dirname(figpath), exist_ok=True)
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(loss_g,label="G")
    plt.plot(loss_d,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'{figpath}.png', bbox_inches='tight', transparent=True)
