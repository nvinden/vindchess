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
        
        self.conv_blocks = list()
        self.skip_convolutions = list()
        self.skip_batch_norm = list()

        for i, conv_depth in enumerate(self.conv_filter_depth):
            if i == len(self.conv_filter_depth) - 1:
                last_depth = self.conv_filter_depth[i]
            else:
                last_depth = self.conv_filter_depth[i + 1]
            
            self.conv_blocks.append(ResidualBlock(depth = conv_depth, n_convolutions = self.n_convs_per_block, last_depth = last_depth))
            self.skip_convolutions.append(nn.Conv2d(self.conv_filter_depth[i], self.conv_filter_depth[min(i + 1, len(self.conv_filter_depth) - 1)], kernel_size=3, padding = 1))
            self.skip_batch_norm.append(nn.BatchNorm2d(self.conv_filter_depth[min(i + 1, len(self.conv_filter_depth) - 1)]))

        # Policy net with a mapping to all possible board moves.
        self.move_policy_net = nn.Linear(self.conv_filter_depth[-1] * 8 * 8, 64 * 64)

        # Policy with board confidence rating
        self.board_confidence = nn.Linear(self.conv_filter_depth[-1] * 8 * 8, 1)

    # Input (6x8x8) state tensor
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype = torch.float32)
        
        x = x.to(torch.float32)
        
        # Does not have a batch
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]

        for conv_block, conv_skip_block in zip(self.conv_blocks, self.skip_convolutions):
            out = conv_block(x)
            x = F.relu(out + conv_skip_block(x))

        board_conf_out = torch.tanh(self.board_confidence(x.view(batch_size, -1))).squeeze(-1)
        move_policy_out = F.softmax(self.move_policy_net(x.view(batch_size, -1)), dim = 1)

        return board_conf_out, move_policy_out
        
class ResidualBlock(nn.Module):
    def __init__(self, depth = 12, n_convolutions = 3, last_depth = None):
        super(ResidualBlock, self).__init__()

        self.depth = depth
        self.n_convolutions = n_convolutions
        self.last_depth = last_depth

        # Creates all CONV2d layers with depth
        if last_depth is None:
            self.conv_layers = [nn.Conv2d(in_channels = depth, out_channels = depth, kernel_size = 3, padding=1) for _ in range(n_convolutions)]

            self.bn_layers = [nn.BatchNorm2d(depth) for _ in range(n_convolutions)]
        else:
            self.conv_layers = [nn.Conv2d(in_channels = depth, out_channels = depth, kernel_size = 3, padding=1) for _ in range(n_convolutions - 1)]
            self.conv_layers.append(nn.Conv2d(in_channels = depth, out_channels = last_depth, kernel_size = 3, padding=1))

            self.bn_layers = [nn.BatchNorm2d(depth) for _ in range(n_convolutions - 1)]
            self.bn_layers.append(nn.BatchNorm2d(last_depth))


    def forward(self, x):
        for i, (conv_l, bn_l) in enumerate(zip(self.conv_layers, self.bn_layers)):
            x = bn_l(conv_l(x))
            if i != self.n_convolutions - 1:
                x = F.relu(x)
        return x