import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, in_shape, action_size, conv_layers=[(16, 8, 4), (32, 4, 2)], dense_layers=[256]):
        """
        Inputs:
            in_shape: (1, C, H, W)
            action_size: int
            conv_layers = List[Tuple(out_channels, kernel_size, stride)]
            dense_layers = List[out_size]
        """
        super(ActorCritic, self).__init__()

        layers = []
        curr_shape = in_shape

        for out_channels, kernel_size, stride in conv_layers:
            layers.append(nn.Conv2d(
                in_channels=curr_shape[1],
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride
            ))

            curr_shape[1] = out_channels

            new_size_func = lambda x: int((x - kernel_size)/stride + 1)
            curr_shape[2] = new_size_func(curr_shape[2])
            curr_shape[3] = new_size_func(curr_shape[3])

        layers.append(nn.Flatten())
        flat_size = curr_shape[1] * curr_shape[2] * curr_shape[3]

        for out_size in dense_layers:
            layers.append(nn.Linear(flat_size, out_size))
            flat_size = out_size
        self.lay = layers
        self.shared = nn.Sequential(*layers)

        self.actor = nn.Linear(flat_size, action_size)
        self.critic = nn.Linear(flat_size, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)