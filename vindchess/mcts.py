from collections import defaultdict

class MCTS():
    def __init__(self):
        self.visited = set()

        # See : https://web.stanford.edu/~surag/posts/alphazero.html
        # for more information on how this search technique works

        self.P = dict()
        self.N = dict()
        self.Q = dict()

    def search(self, state, game, net):
        if game.game_ended(state): 
            return -game.game_reward(state)

        state_key = game.state_to_key(state)

        # This node has not been explored yet
        if state_key not in self.visited:
            self.visited.add(state_key)
            v, self.P[state_key] = net(state)
            return -v

        max_u, best_a = -float("inf"), -1
        valid_moves = game.get_available_moves(state)
        for move in valid_moves:
            s = state_key
            a = move[0] + move[1] # String action encoding of the potential move
            u = 2

        return -v
