from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, input_shape, hidden_dim):
        super(Autoencoder, self).__init__()

        liner_dim = 128 * input_shape[1] // 8 * input_shape[2] // 8

        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(liner_dim, hidden_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, liner_dim),
            nn.ReLU(),
            nn.Unflatten(1, (128, input_shape[1] // 8, input_shape[2] // 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_shape[0], kernel_size=5, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
