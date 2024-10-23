from torch import nn, Tensor
from torch.optim import Optimizer
import torch


class Autoencoder(nn.Module):
    def __init__(self, input_shape: dict, hidden_dim: int):
        super(Autoencoder, self).__init__()

        self._encoder_conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        conv_out_shape = self._calc_out_conv_shape(input_shape)

        liner_dim = conv_out_shape[0] * conv_out_shape[1] * conv_out_shape[2]

        self.encoder = nn.Sequential(
            self._encoder_conv,
            nn.Flatten(),
            nn.Linear(liner_dim, hidden_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, liner_dim),
            nn.ReLU(),
            nn.Unflatten(1, conv_out_shape),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, input_shape[0], kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(input_shape[0], input_shape[0], kernel_size=1),
            nn.Sigmoid(),
        )

    def _calc_out_conv_shape(self, input_shape: dict):
        x = torch.rand((1, *input_shape))
        x = self._encoder_conv(x)
        return x.shape[1:]

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def train_on_batch(
        self, state_batch: Tensor, optimizer: Optimizer, loss_fn: nn.Module
    ):
        self.train()
        decoded = self(state_batch)

        loss = loss_fn(decoded, state_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.eval()

        return loss.item()
