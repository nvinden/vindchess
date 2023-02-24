from vindchess.game import GameManager, Game
from vindchess.model import ChessNet
from vindchess.config import TRAIN_CONFIG
from vindchess.mcts import MCTS

game = Game()

def policy_iter(original_net : ChessNet) -> ChessNet:
    examples = []

    for i in range(TRAIN_CONFIG["number_of_iters"]):
        for e in range(TRAIN_CONFIG["number_of_episodes"]):
            examples += execute_episode(original_net)
        
def execute_episode(net : ChessNet) -> list:
    examples = []
    s = game.create_initial_board()
    mcts = MCTS()

    for _ in range(TRAIN_CONFIG["number_MCTS_sims"]):
            mcts.search(s, game, net)

    return examples

def main():
    base_net = ChessNet()
    policy_iter(base_net)

if __name__ == "__main__":
    main()