import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CHESSNET_CONFIG

def load_policy_net():
    pass

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()

        self.n_convs_per_block = CHESSNET_CONFIG["n_convolutions_per_block"]
        self.conv_filter_depth = CHESSNET_CONFIG["convolutional_filter_depth"]

        self.conv_blocks = [ResidualBlock(self.n_convs_per_block, depth) for depth in self.conv_filter_depth]

        # Policy net with a mapping to all possible board moves.
        self.move_policy_net = nn.Linear(self.conv_filter_depth[-1] * 8 * 8, 64 * 64)

        # Policy with board confidence rating
        self.board_confidence = nn.Linear(self.conv_filter_depth[-1] * 8 * 8, 1)

    def forward(self, x):
        for conv_block in self.conv_blocks:
            x += conv_block(x)

        board_conf_out = F.tanh(self.board_confidence(x))
        move_policy_out = F.softmax(self.board_confidence(x))

        return board_conf_out, move_policy_out
        
class ResidualBlock(nn.Module):
    def __init__(self, depth = 12, n_convolutions = 3):
        super(ResidualBlock, self).__init__()

        self.conv_layers = [nn.Conv2d(in_channels = depth, out_channels = depth, kernel_size = 3, padding=1) for _ in range(n_convolutions)]
        self.bn_layers = [nn.BatchNorm2d(depth) for _ in range(n_convolutions)]

    def forward(self, x):
        for conv_l, bn_l in zip(self.conv_layers, self.bn_layers):
            x = F.relu(bn_l(conv_l(x)))
        return x