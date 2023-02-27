import random
import copy

import torch
import torch.nn as nn

from vindchess.game import GameManager, Game
from vindchess.model import ChessNet
from vindchess.config import TRAIN_CONFIG
from vindchess.mcts import MCTS

game = Game()

DEVICE = "cpu"

def criterion(gt_v, gt_policy, pred_v, pred_policy):
    v_loss = torch.square(gt_v - pred_v)[0]
    policy_loss = torch.dot(gt_policy, torch.log(pred_policy))
    loss = v_loss - policy_loss
    #print(v_loss, policy_loss)
    return loss

def train_chessnet(examples : list, net : ChessNet) -> ChessNet:
    avg_loss = 0
    net = copy.deepcopy(net)

    optimizer = torch.optim.Adam(net.parameters(), lr = TRAIN_CONFIG["lr"])

    for e in range(TRAIN_CONFIG["number_of_epochs"]):
        optimizer.zero_grad()

        random.shuffle(examples)

        # Calculating loss for one complete game
        for game in examples:
            loss = torch.tensor(0.0)
            for move in game:
                state, gt_policy, gt_v = move
                pred_v, pred_policy = net(state)

                pred_policy = pred_policy.squeeze(0)

                gt_policy = torch.tensor(gt_policy, device = DEVICE)
                gt_v = torch.tensor(gt_v, device = DEVICE)
            
                loss += criterion(
                    gt_v = gt_v,
                    gt_policy = gt_policy,
                    pred_v = pred_v,
                    pred_policy = pred_policy
                )
            
            loss /= len(game)

            print(loss)

            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

    avg_loss /= TRAIN_CONFIG["number_of_epochs"]

    return net, avg_loss

def policy_iter(original_net : ChessNet) -> ChessNet:
    examples = []

    nnet = original_net

    for i in range(TRAIN_CONFIG["number_of_iters"]):
        for e in range(TRAIN_CONFIG["number_of_episodes"]):
            examples.append(execute_episode(nnet))
        
        trained_nnet, avg_loss = train_chessnet(examples, original_net)

        nnet = trained_nnet

        print(f"avg_loss: {avg_loss}")

# TODO: put this on the gpu.    
def execute_episode(net : ChessNet) -> list:
    mcts = MCTS()

    examples = []
    s = game.create_initial_board()

    # This determines which net starts as white
    # P1 -> Training Net
    # P2 -> Previous Net
    current_player = 1 if random.uniform(0, 1) >= 0.5 else 2

    while True:
        for _ in range(TRAIN_CONFIG["number_MCTS_sims"]):
            #print("New MCTS")
            mcts.search(s, game, net)

        state_key = game.state_to_key(s)

        pi = mcts.P[state_key].tolist()

        examples.append([s, pi, None])

        a_index = random.choices(list(range(8 ** 4)), weights = pi)[0]
        a_chess = game.q_index_to_chess_encoding(a_index)

        # Illegal move found
        if a_chess not in game.get_available_moves(s):
            #print("Illegal_move")

            examples = assign_rewards(examples, -1.0)
            return examples

        s = game.next_state(s, a_chess)

        print(game.print_state(s))

        if game.game_ended(s):
            examples = assign_rewards(examples, game.game_reward(s))
            return examples
        
        # Changing the orientaton of the board
        s = game.get_flipped_board(s)

def assign_rewards(examples, value):
    if isinstance(value, list):
        raise Exception()
    elif isinstance(value, float):
        return [[x[0], x[1], value] for x in examples]
    else: raise Exception("Value must be type float or list")

def main():
    base_net = ChessNet()
    policy_iter(base_net)

if __name__ == "__main__":
    main()