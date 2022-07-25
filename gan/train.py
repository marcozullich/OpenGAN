import torch
from torch.utils.data import DataLoader
from typing import Union
import os

from .utils import EmptyLoader

def label_smoothing(
    labels:torch.Tensor,
    label_smoothing:float,
):
    if label_smoothing == 0:
        return labels
    else:
        smoother = torch.rand(labels.shape, device=labels.device) * label_smoothing
        return labels * (1 - smoother) + (1 - labels) * smoother

def discriminator_real_data(
    discriminator:torch.nn.Module,
    batch_data:torch.Tensor,
    label_smoothing_factor:float,
    loss_discriminator:torch.nn.Module,
):
    discriminator.zero_grad()
    batch_size = batch_data.shape[0]
    label = torch.full((batch_size,), 1, device=batch_data.device, dtype=torch.float)
    label = label_smoothing(label, label_smoothing_factor)
    # print(batch_data.shape)
    output = discriminator(batch_data).view(-1)
    error_discrim_real = loss_discriminator(output, label)
    error_discrim_real.backward()
    D_x = output.mean().item()
    return error_discrim_real, D_x
    
def discriminator_fake_data(
    discriminator:torch.nn.Module,
    generator:torch.nn.Module,
    noise:torch.Tensor,
    label_smoothing_factor:float,
    loss_discriminator:torch.nn.Module,
):
    discriminator.zero_grad()
    batch_size = noise.shape[0]
    fake_data = generator(noise)
    # print(fake_data.shape)
    label = torch.full((batch_size,), 0, device=noise.device, dtype=torch.float)
    label = label_smoothing(label, label_smoothing_factor)
    output = discriminator(fake_data.detach()).view(-1)
    error_discrim_fake = loss_discriminator(output, label)
    error_discrim_fake.backward()
    D_G_z1 = output.mean().item()
    return error_discrim_fake, D_G_z1, fake_data

def discriminator_ood_data(
    discriminator:torch.nn.Module,
    ood_data:torch.Tensor,
    label_smoothing_factor:float,
    loss_discriminator:torch.nn.Module,
    lambda_O:float,
):
    discriminator.zero_grad()
    batch_size = ood_data.shape[0]
    label = torch.full((batch_size,), 0, device=ood_data.device, dtype=torch.float)
    label = label_smoothing(label, label_smoothing_factor)
    output = discriminator(ood_data).view(-1)
    error_discrim_ood = loss_discriminator(output, label) * lambda_O
    error_discrim_ood.backward()
    D_x_ood = output.mean().item()
    return error_discrim_ood, D_x_ood


def generator_train(
    generator:torch.nn.Module,
    discriminator:torch.nn.Module,
    label_smoothing_factor:float,
    fake_data:torch.Tensor,
    loss_generator:torch.nn.Module,
    optimizer_generator:torch.optim.Optimizer,
    lambda_G:float,
):
    generator.zero_grad()
    label = torch.full((fake_data.shape[0],), 1, device=fake_data.device)
    label = label_smoothing(label, label_smoothing_factor)
    output = discriminator(fake_data).view(-1)
    error_gen = loss_generator(output, label) * lambda_G
    error_gen.backward()
    optimizer_generator.step()
    D_G_z2 = output.mean().item()
    return error_gen, D_G_z2

def train(
    generator:torch.nn.Module,
    discriminator:torch.nn.Module,
    optimizer_g:torch.optim.Optimizer,
    optimizer_d:torch.optim.Optimizer,
    trainloader:DataLoader,
    epochs:int,
    device:Union[torch.device, str],
    label_smoothing_factor:float,
    loss_fn:torch.nn.Module,
    latent_dim:int,
    ite_print:int=None,
    oodloader:DataLoader=None,
    lambda_O:float=0.0,
    lambda_G:float=1.0,
    save_path:str=None
):
    discriminator.train()
    discriminator.to(device)
    generator.train()
    generator.to(device)
    G_losses = []
    D_losses = []

    if oodloader is None:
        oodloader = EmptyLoader()

    if ite_print is None:
        ite_print = len(trainloader)

    for epoch in range(epochs):
        for i, (data, ooddata) in enumerate(zip(trainloader, oodloader)):
            data = data.to(device)
            noise = torch.randn(data.shape[0], latent_dim, 8, 8, device=device)
            # Discriminator
            error_discrim_real, D_x = discriminator_real_data(discriminator, data, label_smoothing_factor, loss_fn)
            error_discrim_fake, D_G_z1, fake_data = discriminator_fake_data(discriminator, generator, noise, label_smoothing_factor, loss_fn)    
            
            # Discriminator OOD if ooddata exists and lambda_O > 0
            D_x_ood = 0
            error_discrim_ood = 0
            if ooddata is not None and lambda_O > 0.0:
                ooddata = ooddata.to(device)
                error_discrim_ood, D_x_ood = discriminator_ood_data(discriminator, ooddata, label_smoothing_factor, loss_fn, lambda_O)

            optimizer_d.step()
            error_discrim = error_discrim_real + error_discrim_fake + error_discrim_ood

            # Generator
            error_gen, D_G_z2 = generator_train(generator, discriminator, label_smoothing_factor, fake_data, loss_fn, optimizer_g, lambda_G)

            if save_path is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                name, ext = os.path.splitext(save_path)
                if ext == "" or ext == ".":
                    ext = ".pt"
                filename_D = f"{name}_D_{epoch+1}{ext}"
                filename_G = f"{name}_G_{epoch+1}{ext}"
                torch.save(discriminator.state_dict(), filename_D)
                torch.save(generator.state_dict(), filename_G)

            G_losses.append(error_gen.item())
            D_losses.append(error_discrim.item())
            if (i+1) % ite_print == 0:
                dxood = f"{D_x_ood:.4f}" if D_x_ood is not None else "--"
                print(f"Ep. {epoch+1}/{epochs}| It. {i+1}/{len(trainloader)} | D_x: {D_x:.4f} | D_x_ood: {dxood} | D_G_z1: {D_G_z1:.4f} | D_G_z2: {D_G_z2:.4f} | ℒ_D: {error_discrim.item():.4f} | ℒ_G: {error_gen.item():.4f}")

    return G_losses, D_losses