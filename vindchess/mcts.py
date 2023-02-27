from math import sqrt
from collections import defaultdict

import numpy as np
import torch

from vindchess.game import Game
from vindchess.config import TRAIN_CONFIG
from vindchess.model import ChessNet



class MCTS():
    def __init__(self):
        self.visited = set()

        # See : https://web.stanford.edu/~surag/posts/alphazero.html
        # for more information on how this search technique works

        self.P = dict()
        self.Q = defaultdict(lambda: torch.zeros([8**4]), dtype = torch.float32)
        self.N = defaultdict(lambda: [0] * 8**4)

    def search(self, state, game : Game, net : ChessNet):
        #print(game.print_state(state))

        if game.game_ended(state): 
            return -game.game_reward(state)

        state_key = game.state_to_key(state)

        # This node has not been explored yet
        if state_key not in self.visited:
            self.visited.add(state_key)
            v, new_P = net(state)
            self.P[state_key] = new_P.squeeze(0)

            return -v

        max_u, best_a = -float("inf"), -1
        valid_moves = game.get_available_moves(state)
        for move in valid_moves:
            s = state_key

            # Getting the index of the q values that correspond to the index of the 
            a = game.chess_encoding_to_q_index(move[0], move[1])

            u = self.Q[s][a] + TRAIN_CONFIG['c_puct']*self.P[s][a]*sqrt(sum(self.N[s]))/(1+self.N[s][a])
            if u>max_u:
                max_u = u
                best_a = a

        a = best_a

        m1, m2 = game.q_index_to_chess_encoding(a)
        sp = game.next_state(state, (m1, m2))

        # Flipping board so MCTS searches from opponents perspective next
        reversed_state = game.get_flipped_board(sp)

        v = self.search(reversed_state, game, net)

        self.Q[s][a] = (self.N[s][a]*self.Q[s][a] + v)/(self.N[s][a]+1)
        self.N[s][a] += 1

        return -v