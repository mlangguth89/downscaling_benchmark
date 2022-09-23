import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from network_unet import UNet
from network_critic_model import Discriminator
import gc
import time

sys.path.append('../')
from main_scripts.dataset_temp_v2 import CustomTemperatureDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def recon_loss(real_data, gen_data):
    # initialize reconstruction loss
    rloss = 0.

    rloss += (torch.abs(gen_data - real_data)).mean()

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
            return lr * torch.exp(decay_rate)
        elif epoch >= decay_end:
            return lr

    return lr_scheduler


class train_WGAN():
    """
    Training WGAN model with gradient penalty
    """

    def __init__(self, generator: nn.Module = None, critic: nn.Module = None, train_dataloader: DataLoader = None
                 , val_dataloader: DataLoader = None, hparams: dict = None):
        """

        """
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.generator = generator.to(device)
        self.critic = critic.to(device)
        self.hparams = hparams

        self.opt_critic = optim.Adam(self.critic.parameters(), lr=self.hparams.lr_critic, betas=(0.0, 0.9))
        self.opt_gen = optim.Adam(self.generator.parameters(), lr=self.hparams.lr_gn, betas=(0.0, 0.9))

        self.best_cr_loss = 1000
        self.best_g_loss = 1000

    def train(self):
        """

        """
        for epoch in range(self.hparams.epochs):

            start = time.time()
            self.generator.train()
            self.critic.train()

            for batch_idx, train_data in enumerate(self.train_dataloader):
                self.generator.train()
                self.critic.train()
                input_data = train_data[0].to(device)
                target_data = train_data[1].to(device)

                # Training the critic model
                for i in range(self.hparams.critic_iterations):

                    generator_output = self.generator(input_data)
                    critic_real = self.critic(target_data)
                    critic_fake = self.critic(generator_output)
                    # NEED TO VALIDATE
                    # =-=-=-=-=-==-
                    gp = self.gradient_penalty(target_data, generator_output, device=device)
                    # =-=-=-=-=-==-

                    loss_critic = (
                            -(torch.mean(critic_real) - torch.mean(critic_fake)) + self.hparams.lambada_gp * gp
                    )

                    self.critic.zero_grad()
                    loss_critic.backward(retain_graph=True)
                    self.opt_critic.step()

                # Training the generator model
                gen_fake = self.critic(generator_output).reshape(-1)
                loss_gen = -torch.mean(gen_fake)
                loss_rec = recon_loss(target_data, generator_output)
                g_loss = loss_gen + self.hparams.recon_weight * loss_rec
                self.generator.zero_grad()
                g_loss.backward()
                self.opt_gen.step()

            end = time.time()
            # Printing the results for each epoch
            gc.collect()
            loss_val_c, loss_val_gen = self.validation()
            print(
                f"Epoch [{epoch+1}/{self.hparams.epochs}] Batch {self.hparams.batch_size}/{len(self.train_dataloader)} \
                  Loss D Train: {loss_critic.item():.4f}, loss G Train: {loss_gen.item():.4f},"
                f"Loss D Val: {loss_val_gen:.4f}, loss G Val: {loss_val_gen:.4f},"
                f", Time per 1 epoch: {end-start:.4f} sec."
            )

            # if self.best_g_loss > loss_val_gen:
            #     self.best_g_loss = loss_val_gen
            #     self.save_checkpoint(epoch=epoch, loss_g=loss_val_gen, loss_cr=loss_val_gen)
            #
            # if self.best_cr_loss > loss_val_gen:
            #     self.best_cr_loss = loss_val_gen

            self.update_lr(self.hparams)  # Updating learning rate

    def update_lr(self, epoch: int = None):
        updater = get_lr_decay()
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
        for batch_idx, train_data in enumerate(self.val_dataloader):
            input_data = train_data[0].to(device)
            target_data = train_data[1].to(device)

            generator_output = self.generator(input_data)
            critic_real = self.critic(target_data)
            critic_fake = self.critic(generator_output)

            # Critic loss calculation
            gp = self.gradient_penalty(target_data, generator_output, device=device)
            loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + self.hparams.lambada_gp * gp
            )

            # Generator loss calculation
            gen_fake = self.critic(generator_output).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            loss_rec = recon_loss(target_data, generator_output)
            g_loss = loss_gen + self.hparams.recon_weight * loss_rec

            loss_c += loss_critic.item()
            loss_g += g_loss.item()

        return loss_c, g_loss

    def save_checkpoint(self, epoch: int = None, loss_g: float = None, loss_cr: float = None):
        """

        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.opt_gen.state_dict(),
            'loss': loss_g,
        }, self.hparams.save_dir)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.opt_critic.state_dict(),
            'loss': loss_cr,
        }, self.hparams.save_dir)

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
    parser.add_argument("--save_dir", type=str, default="C:\\Users\\max_b\\PycharmProjects\\downscaling_maelstrom",
                        help="The output directory")
    parser.add_argument("--epochs", type=int, default=2, help="The checkpoint directory")
    parser.add_argument("--batch_size", type=int, default=32, help="The checkpoint directory")
    parser.add_argument("--critic_iterations", type=float, default=1, help="The checkpoint directory")
    parser.add_argument("--lr_gn", type=float, default=5.e-05, help="The checkpoint directory")
    parser.add_argument("--lr_gn_end", type=float, default=5.e-06, help="The checkpoint directory")
    parser.add_argument("--lr_critic", type=float, default=1.e-06, help="The checkpoint directory")
    parser.add_argument("--decay_start", type=int, default=5, help="The checkpoint directory")
    parser.add_argument("--decay_end", type=int, default=10, help="The checkpoint directory")
    parser.add_argument("--lambada_gp", type=float, default=10, help="The checkpoint directory")
    parser.add_argument("--recon_weight", type=float, default=0.2, help="The checkpoint directory")

    parser.add_argument("--checkpoint_dir", type=str, required=False,
                        default="C:\\Users\\max_b\\PycharmProjects\\downscaling_maelstrom",
                        help="Please provide the checkpoint directory")

    args = parser.parse_args()

    critic_model = Discriminator((1, 120, 96))
    generator_model = UNet(n_channels=9)

    data_train = CustomTemperatureDataset(file_path=args.train_dir)
    train_dataloader = DataLoader(data_train, batch_size=args.batch_size, shuffle=False)

    data_val = CustomTemperatureDataset(file_path=args.val_dir)
    val_dataloader = DataLoader(data_val, batch_size=args.batch_size, shuffle=False)

    WGAN = train_WGAN(generator=generator_model, critic=critic_model, train_dataloader=train_dataloader,
                      val_dataloader=val_dataloader, hparams=args)
    WGAN.train()


if __name__ == "__main__":
    run()
