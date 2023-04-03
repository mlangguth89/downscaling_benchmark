__email__ = "maximbr@post.bgu.ac.il"
__author__ = "Maxim Bragilovski"
__date__ = "2022-12-08"

import sys
sys.path.append('../')

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from models.network_unet import UNet
from models.network_critic import Discriminator
from torch.autograd import Variable
import torch.autograd as autograd
import math
import os
from main_scripts.dataset_temp import CustomTemperatureDataset
from torch.optim import lr_scheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def recon_loss(real_data, gen_data):
    # initialize reconstruction loss
    b_loss = torch.mean(torch.abs(gen_data - real_data))
    rloss = torch.abs(gen_data - real_data).mean()
    return rloss


def get_lr_decay(hparams: dict = None):
    """
    Get callable of learning rate scheduler which can be used as callabck in Keras models.
    Exponential decay is applied to change the learning rate from the start to the end value.
    Note that the exponential decay is calculated based on the learning rate of the generator, but applies to both.
    :return: learning rate scheduler
    """
    decay_st, decay_end = hparams.decay_start, hparams.decay_end
    lr_start, lr_end = hparams.lr_gn, hparams.lr_gn_end

    if not decay_end > decay_st:
        raise ValueError("Epoch for end of learning rate decay must be large than start epoch. " +
                         "Your values: {0:d}, {1:d})".format(decay_st, decay_end))

    ne_decay = decay_end - decay_st
    # calculate decay rate from start and end learning rate
    decay_rate = 1. / ne_decay * np.log(lr_end / lr_start)

    def lr_scheduler(epoch, lr):
        if epoch < decay_st:
            return lr
        elif decay_st <= epoch < decay_end:
            return lr * torch.exp(torch.from_numpy(np.array(decay_rate)))
        elif epoch >= decay_end:
            return lr

    return lr_scheduler


class BuildWGANModel:
    """
    Training WGAN model with gradient penalty
    """

    def __init__(self, generator: nn.Module = None,
                 critic: nn.Module = None,
                 train_loader: DataLoader = None,
                 val_loader: DataLoader = None,
                 hparams: dict = None,
                 checkpoint_save: int = None,
                 save_dir: str = None,
                 dataset_type: str = 'precipitation'):
        """

        """
        self.dataset_type = dataset_type
        self.train_dataloader = train_loader
        self.val_dataloader = val_loader
        self.generator = generator.to(device)
        self.critic = critic.to(device)
        self.hparams = hparams
        self.checkpoint_save = checkpoint_save
        self.save_dir = save_dir
        self.G_optimizer_lr = self.hparams.lr_gn
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=self.hparams.lr_critic, betas=(0.5, 0.999))
        self.opt_gen = optim.Adam(self.generator.parameters(), lr=self.hparams.lr_gn, betas=(0.5, 0.999))

        self.scheduler_critic = lr_scheduler.ReduceLROnPlateau(self.opt_critic)
        self.scheduler_gen = lr_scheduler.ReduceLROnPlateau(self.opt_gen)

        self.best_cr_loss = 1000
        self.best_g_loss = 1000

    def fit(self):
        """

        """
        jj = 0
        current_step = 0
        # max_iterations = math.ceil(len(self.train_dataloader.dataset) / self.hparams.batch_size)
        print(f' size of dataset {self.train_dataloader.dataset.n_samples} batch size {self.hparams.batch_size}' )
        print(f' iterations {self.train_dataloader.dataset.n_samples / self.hparams.batch_size}')
        max_iterations = math.floor(self.train_dataloader.dataset.n_samples / self.hparams.batch_size)
        counter_2 = 0
        loss_train_rec = 0
        jj = 0

        for epoch in range(self.hparams.epochs):

            if jj == self.hparams.critic_iterations:
                jj = 0
            val_loss = 0
            for i, train_data in enumerate(self.train_dataloader):

                current_step += 1
                self.generator.train()
                self.critic.train()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.opt_critic.zero_grad()

                if self.dataset_type == 'precipitation':
                    input_data = train_data['L'].to(device)
                    target_data = train_data['H'][:, None].to(device)
                else:
                    input_data = train_data['L'].to(device)
                    target_data = train_data['H'].to(device)

                generator_output = self.generator(input_data)
                critic_real = self.critic(target_data)
                critic_fake = self.critic(generator_output)

                gp = self.compute_gradient_penalty(target_data,
                                                   generator_output,
                                                   device=device)
                # check = self.gradient_penalty(target_data, generator_output, device=device)
                loss_critic = -torch.mean(critic_real) + torch.mean(critic_fake) + self.hparams.lambada_gp * gp

                loss_critic.backward()
                self.opt_critic.step()

                self.opt_gen.zero_grad()

                # -----------------
                #  Train Generator
                # -----------------
                if jj == (i % self.hparams.critic_iterations):

                    generator_output = self.generator(input_data)

                    critic_fake = self.critic(generator_output)

                    loss_gen = -torch.mean(critic_fake)
                    loss_rec = recon_loss(target_data, generator_output)

                    g_loss = loss_gen + self.hparams.recon_weight * loss_rec

                    g_loss.backward()
                    self.opt_gen.step()

                    val_loss = loss_rec.item()

                    # print(
                    #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Rec loss: %f]"
                    #     % (epoch, self.hparams.epochs, i, len(self.train_dataloader), loss_critic.item(), g_loss.item(),
                    #        loss_rec.item())
                    # )

            print('++++++++++++++++++++++++++++++++')
            print(
                "[Epoch %d/%d] [Batch %d] [D loss: %f] [G loss: %f] [Rec loss: %f]"
                % (epoch, self.hparams.epochs, i, loss_critic.item(), g_loss.item(),
                   val_loss)
            )

            loss_val_c, loss_val_gen, loss_rec, count_1, lr_g, lr_c = self.validation()
            self.save_checkpoint(epoch=epoch, step=current_step)

            print(
                "[Epoch %d/%d] [D Val loss: %f] [G loss: %f] [Rec loss: %f] [LR C: %f] [LR G: %f]"
                % (epoch, self.hparams.epochs, loss_val_c / count_1, loss_val_gen / count_1, loss_rec / count_1,
                   lr_c, lr_g)
            )

            print('++++++++++++++++++++++++++++++++')

            self.scheduler_critic.step(loss_rec / count_1)
            self.scheduler_gen.step(loss_rec / count_1)

            jj += 1


    def update_lr(self, epoch: int = None):
        updater = get_lr_decay(self.hparams)
        new_lr_c = updater(epoch, self.hparams.lr_critic)
        new_lr_g = updater(epoch, self.hparams.lr_gn)

        for g in self.opt_critic.param_groups:
            g['lr'] = new_lr_c

        self.hparams.lr_critic = new_lr_c

        for g in self.opt_gen.param_groups:
            g['lr'] = new_lr_g

    def validation(self):
        """

        """
        self.generator.eval()
        self.critic.eval()
        loss_c = 0
        loss_g = 0
        loss_r = 0
        count = 0
        for batch_idx, train_data in enumerate(self.val_dataloader):
            count += 1
            if self.dataset_type == 'precipitation':
                input_data = train_data['L'].to(device)
                target_data = train_data['H'][:, None].to(device)
            else:
                input_data = train_data['L'].to(device)
                target_data = train_data['H'].to(device)

            target_data = target_data.to(device)

            generator_output = self.generator(input_data)
            critic_real = self.critic(target_data)
            critic_fake = self.critic(generator_output)

            # Critic loss calculation
            gp = self.compute_gradient_penalty(target_data, generator_output, device=device)
            loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + self.hparams.lambada_gp * gp
            )

            check = self.gradient_penalty(target_data, generator_output, device=device)

            # Generator loss calculation
            gen_fake = self.critic(generator_output).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            loss_rec = recon_loss(target_data, generator_output)
            g_loss = loss_gen + self.hparams.recon_weight * loss_rec

            loss_c += loss_critic.item()
            loss_g += g_loss.item()
            loss_r += loss_rec.item()

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        lr_g = get_lr(self.opt_gen)
        lr_c = get_lr(self.opt_critic)

        return loss_c, g_loss, loss_r, count, lr_g, lr_c

    def save_checkpoint(self, epoch: int = None, loss_g: float = None, loss_cr: float = None, step: int = None):
        """

        """

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.opt_gen.state_dict(),
            'loss': loss_g,
        }, os.path.join(self.hparams.save_dir, f'generator_step{step}.pth'))

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.opt_critic.state_dict(),
            'loss': loss_cr,
        }, os.path.join(self.hparams.save_dir, f'critic_step{step}.pth'))

    def gradient_penalty(self, real, fake, device="cpu"):
        """

        """
        BATCH_SIZE, C, H, W = real.shape
        alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
        interpolated_images = real * alpha + fake * (1 - alpha)

        # Calculate critic scores
        mixed_scores = self.critic(interpolated_images)

        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        return gradient_penalty

    # Taken from: https://github.com/eriklindernoren/PyTorch-GAN/blob/36d3c77e5ff20ebe0aeefd322326a134a279b93e/implementations/wgan_gp/wgan_gp.py#L119
    def compute_gradient_penalty(self, real_samples, fake_samples, device="cpu"):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.critic(interpolates)
        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=False,
                        default="C:\\Users\\max_b\\PycharmProjects\\downscaling_maelstrom\\preproc_era5_crea6_small.nc",
                        # C:\\Users\\max_b\\PycharmProjects\\downscaling_maelstrom\\preproc_era5_crea6_small.nc
                        help="The directory where training data (.nc files) are stored")
    parser.add_argument("--val_dir", type=str, required=False,
                        default="C:\\Users\\max_b\\PycharmProjects\\downscaling_maelstrom\\preproc_era5_crea6_small.nc",
                        # C:\\Users\\max_b\\PycharmProjects\\downscaling_maelstrom\\preproc_era5_crea6_small.nc
                        help="The directory where test data (.nc files) are stored")
    parser.add_argument("--save_dir", type=str, default="C:\\Users\\max_b\\PycharmProjects\\downscaling_maelstrom\\output\\unet",
                        help="The output directory")
    parser.add_argument("--epochs", type=int, default=250, help="The checkpoint directory")
    parser.add_argument("--batch_size", type=int, default=16, help="The checkpoint directory")
    parser.add_argument("--critic_iterations", type=float, default=4, help="The checkpoint directory")
    parser.add_argument("--lr_gn", type=float, default=5.e-05, help="The checkpoint directory")
    parser.add_argument("--lr_gn_end", type=float, default=5.e-06, help="The checkpoint directory")
    parser.add_argument("--lr_critic", type=float, default=1.e-06, help="The checkpoint directory")
    parser.add_argument("--decay_start", type=int, default=25, help="The checkpoint directory")
    parser.add_argument("--decay_end", type=int, default=50, help="The checkpoint directory")
    parser.add_argument("--lambada_gp", type=float, default=10, help="The checkpoint directory")
    parser.add_argument("--recon_weight", type=float, default=1000, help="The checkpoint directory")

    parser.add_argument("--checkpoint_dir", type=str, required=False,
                        default="C:\\Users\\max_b\\PycharmProjects\\downscaling_maelstrom\\output\\unet",
                        help="Please provide the checkpoint directory")

    args = parser.parse_args()

    critic_model = Discriminator((1, 120, 96))
    generator_model = UNet(n_channels=9)

    data_train = CustomTemperatureDataset(file_path=args.train_dir)
    train_dataloader = DataLoader(data_train, batch_size=args.batch_size, shuffle=False)

    data_val = CustomTemperatureDataset(file_path=args.val_dir)
    val_dataloader = DataLoader(data_val, batch_size=args.batch_size, shuffle=False)

    WGAN = BuildWGANModel(generator=generator_model, critic=critic_model, train_dataloader=train_dataloader,
                      val_dataloader=val_dataloader, hparams=args)
    WGAN.fit()


if __name__ == "__main__":
    run()

