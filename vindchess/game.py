from .model import ChessNet

import random
import numpy as np
import torch

from itertools import product

random.seed = 41

PIECE_TO_INDEX = {"p": 0, "k": 1, "b": 2, "r": 3, "Q": 4, "K": 5}
INDEX_TO_PIECE = {val: key for key, val in PIECE_TO_INDEX.items()}

ALPH_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7}

GET_OPPONENT_COLOUR = {"W": "B", "B": "W"}

ALL_STRATEGIES = ["vindgod", "random", "simple_heuristic"]

def rc2str(row, column):
    column_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
    return column_letters[column] + str(row + 1)

def str2rc(position):
    return (int(position[1]) - 1, int(ALPH_TO_INDEX[position[0]]))

def on_board(row, column):
    return row >= 0 and row <= 7 and column >= 0 and column <= 7

class GameManager():
    def __init__(self):
        self.game = Game()

        self.policy_net = ChessNet()

    def play_game(self, p1_strategy : str, p2_strategy, verbose = True):
        assert p1_strategy in ALL_STRATEGIES
        assert p2_strategy in ALL_STRATEGIES

        turn = 1 if self.game.p1_colour == "W" else 2

        if verbose:
                print(self.game)

        while(1):
            # See available actions, make a move
            if turn == 1:
                p1_available_moves = self.game.get_available_moves(player = 1)
                selected_move = self.select_move(p1_available_moves, p1_strategy, 1)
                self.make_move(selected_move)
            elif turn == 2:
                p2_available_moves = self.game.get_available_moves(player = 2)
                selected_move = self.select_move(p2_available_moves, p2_strategy, 2)
                self.make_move(selected_move)

            # TODO: Make sure to check that the selected move is indeed valid

            if verbose:
                print("Move made " + str(selected_move))
                print(self.game)

            outcome = self.state_condition()
            if outcome in ["W", "B", "T"]:
                break

            # Change whos turn it is
            turn = 2 if turn == 1 else 1

    # Returns either ["W", "B", "T", "C"]
    # W: White wins
    # B: Black wins
    # T: Tied game
    # C: Continue
    def state_condition(self):
        # If P1 has no king
        if 1 not in np.unique(self.game.board[-1]):
            return "W" if self.game.p1_colour == "W" else "B"

        # If P2 has no king
        if -1 not in np.unique(self.game.board[-1]):
            return "B" if self.game.p1_colour == "W" else "W"

        return "C"
        
    def make_move(self, move):
        rc1 = str2rc(move[0])
        rc2 = str2rc(move[1])

        self.game.board[:, rc2[0], rc2[1]] = self.game.board[:, rc1[0], rc1[1]]
        self.game.board[:, rc1[0], rc1[1]] = 0

    def select_move(self, available_moves, strategy, player_number):
        if strategy == "random":
            return random.choice(available_moves)
        elif strategy == "simple_heuristics":
            # TODO: These two
            raise NotImplementedError
        elif strategy == "vindgod":
            board_float = self.game.board.astype("float32")
            board_float = np.expand_dims(board_float, 0)
            _, move_matrix = self.policy_net(board_float)

            # TODO: Instead of just returning top move, implement a search to
            # do this instead using some sort of strategy with beta-pruning and
            # searching for best board period

            best_move_index = torch.argmax(move_matrix).item()

            start, end = self.game.q_index_to_chess_encoding(best_move_index)

            return (start, end)

    def train_game(self):
        pass

class Game():
    knight_moves = [(-2, -1), (-2, 1), (-1, 2), (1, 2), (2, -1), (2, 1), (-1, -2), (1, -2)]

    def __init__(self, p1_colour = None):
        self.p1_colour = p1_colour

        if p1_colour == None:
            self.p1_colour = "B" if random.uniform(0, 1) > 0.5 else "W"

        assert self.p1_colour in ["B", "W"]

        self.board = self.create_initial_board()

  # Returns a list of valud moves for the player
    # ex: ["a4", "a5"]
    def get_available_moves_deprecated(self, player) -> list:
        assert player in [1, 2]

        if player == 1:
            board = self.board
            available_actions = self.__get_available_moves(board)
            return available_actions
        elif player == 2:
            board = self.__get_flipped_board(self.board)
            available_actions = self.__get_available_moves(board)
            available_actions =  self.__get_flipped_actions(available_actions)
            return available_actions 
    
    def __get_flipped_actions(self, actions):
        out = []
        for pos_start, pos_end in actions:
            start_row = 9 - int(pos_start[1])
            end_row = 9 - int(pos_end[1])
            out.append((pos_start[0] + str(start_row), pos_end[0] + str(end_row)))

        return out

    def get_flipped_board(self, board : np.ndarray):
        board = board.copy()

        board = board * -1
        board = np.flip(board, axis = 1).copy()
        board = np.flip(board, axis = 2).copy()

        return board

    # Internal function 
    def __iteratively_get_linear_translations(self, board, initial_position : tuple, transformation : tuple):
        assert len(initial_position) == 2
        assert len(transformation) == 2

        x_start, y_start = initial_position
        x_trans, y_trans = transformation

        translations = []

        for i in range(1, 7):
            # Proposed move position
            x_curr = i * x_trans + x_start
            y_curr = i * y_trans + y_start

            # Make sure new move is in bounds
            if not on_board(x_curr, y_curr):
                break

            # If runs into own piece    
            if board[x_curr, y_curr] == 1:
                break

            # If runs into enemy    
            if board[x_curr, y_curr] == -1:
                translations.append((rc2str(x_start, y_start), rc2str(x_curr, y_curr)))
                break

            translations.append((rc2str(x_start, y_start), rc2str(x_curr, y_curr)))

        return translations
        
    def get_available_moves(self, state):
        board = state.copy()

        moves = []

        # All pieces
        compressed_board = board.sum(0)
        p1_pieces = np.where(compressed_board == 1, 1, 0)
        p2_pieces = np.where(compressed_board == -1, 1, 0)
        all_pieces = np.absolute(compressed_board)

        # Getting availble pawn moves
        p1_pawns = np.where(board[0] == 1, 1, 0)
        pawn_positions = np.nonzero(p1_pawns)

        for (r, c) in zip(pawn_positions[0], pawn_positions[1]):
            # Can move one position forwards?
            if on_board(r-1, c) and all_pieces[r - 1, c] == 0:
                moves.append((rc2str(r, c), rc2str(r - 1, c)))

            # Can move two on first move?
            if all_pieces[r - 2, c] == 0 and all_pieces[r - 1, c] == 0 and r == 6:
                moves.append((rc2str(r, c), rc2str(r - 2, c)))

            # Can a pawn take left?
            if on_board(r - 1, c - 1) and p2_pieces[r - 1, c - 1] == 1:
                moves.append((rc2str(r, c), rc2str(r - 1, c - 1)))

            # Can a pawn take right?
            if on_board(r - 1, c + 1) and p2_pieces[r - 1, c + 1] == 1:
                moves.append((rc2str(r, c), rc2str(r - 1, c + 1)))

        # Getting availble knight moves
        p1_knights = np.where(board[1] == 1, 1, 0)
        knight_positions = np.nonzero(p1_knights)

        for (r, c) in zip(knight_positions[0], knight_positions[1]):
            for r_k, c_k in self.knight_moves:
                if on_board(r + r_k, c + c_k) and p1_pieces[r + r_k, c + c_k] == 0:
                    moves.append((rc2str(r, c), rc2str(r + r_k, c + c_k)))        

        # Getting availble bishop moves
        p1_bishop = np.where(board[2] == 1, 1, 0)
        bishop_positions = np.nonzero(p1_bishop)

        for (r, c) in zip(bishop_positions[0], bishop_positions[1]):
            # All diagonol movements:
            nw_movements = self.__iteratively_get_linear_translations(compressed_board, (r, c), (-1, -1))
            ne_movements = self.__iteratively_get_linear_translations(compressed_board, (r, c), (-1,  1))
            sw_movements = self.__iteratively_get_linear_translations(compressed_board, (r, c), ( 1, -1))
            we_movements = self.__iteratively_get_linear_translations(compressed_board, (r, c), ( 1,  1))

            moves.extend(nw_movements)
            moves.extend(ne_movements)
            moves.extend(sw_movements)
            moves.extend(we_movements)
            
        # Getting availble rook moves
        p1_rook = np.where(board[3] == 1, 1, 0)
        rook_positions = np.nonzero(p1_rook)

        for (r, c) in zip(rook_positions[0], rook_positions[1]):
            # All horizontal and vertical movements:
            n_movements = self.__iteratively_get_linear_translations(compressed_board, (r, c), ( -1, 0))
            s_movements = self.__iteratively_get_linear_translations(compressed_board, (r, c), ( 1, 0))
            e_movements = self.__iteratively_get_linear_translations(compressed_board, (r, c), ( 0, 1))
            w_movements = self.__iteratively_get_linear_translations(compressed_board, (r, c), ( 0, -1))

            moves.extend(n_movements)
            moves.extend(s_movements)
            moves.extend(e_movements)
            moves.extend(w_movements)

        # Getting availble Queen moves
        p1_queen = np.where(board[4] == 1, 1, 0)
        queen_positions = np.nonzero(p1_queen)

        for (r, c) in zip(queen_positions[0], queen_positions[1]):
            # All diagonal moves
            nw_movements = self.__iteratively_get_linear_translations(compressed_board, (r, c), (-1, -1))
            ne_movements = self.__iteratively_get_linear_translations(compressed_board, (r, c), (-1,  1))
            sw_movements = self.__iteratively_get_linear_translations(compressed_board, (r, c), ( 1, -1))
            we_movements = self.__iteratively_get_linear_translations(compressed_board, (r, c), ( 1,  1))

            # All horizontal/vertical movements:
            n_movements = self.__iteratively_get_linear_translations(compressed_board, (r, c), ( -1, 0))
            s_movements = self.__iteratively_get_linear_translations(compressed_board, (r, c), ( 1, 0))
            e_movements = self.__iteratively_get_linear_translations(compressed_board, (r, c), ( 0, 1))
            w_movements = self.__iteratively_get_linear_translations(compressed_board, (r, c), ( 0, -1))

            moves.extend(nw_movements)
            moves.extend(ne_movements)
            moves.extend(sw_movements)
            moves.extend(we_movements)

            moves.extend(n_movements)
            moves.extend(s_movements)
            moves.extend(e_movements)
            moves.extend(w_movements)

        # Getting availble King moves
        p1_king = np.where(board[5] == 1, 1, 0)
        king_x, king_y = np.nonzero(p1_king)
        king_x = king_x.item()
        king_y = king_y.item()

        for (king_x_trans, king_y_trans) in product([-1, 0, 1], [-1, 0, 1]):
            if king_x_trans == 0 and king_y_trans == 0:
                continue

            king_new_x_pos = king_x + king_x_trans
            king_new_y_pos = king_y + king_y_trans

            if not on_board(king_new_x_pos, king_new_y_pos):
                continue

            if compressed_board[king_new_x_pos, king_new_y_pos] == 1:
                continue

            moves.append((rc2str(king_x, king_y), rc2str(king_new_x_pos, king_new_y_pos)))

        return moves

    def __get_available_p2_moves(self):
        pass

    def get_trad_board(self, state):
        board = state.copy()
        non_zero_positions = np.nonzero(board)

        board_out = [[".."] * 8 for _ in range(8)]

        for i, j, k in zip(non_zero_positions[0], non_zero_positions[1], non_zero_positions[2]):
            if board[i, j, k] == 0:
                continue
            colour = self.p1_colour if board[i, j, k] > 0 else GET_OPPONENT_COLOUR[self.p1_colour]
            piece = INDEX_TO_PIECE[i]

            board_out[j][k] = colour + piece
        
        return board_out

    def create_initial_board(self):
        board = np.zeros(shape = (6, 8, 8), dtype = np.int8)

        # PAWNS
        board[0] = np.array([[ 0,  0,  0,  0,  0,  0,  0,  0],
                            [-1, -1, -1, -1, -1, -1, -1, -1],
                            [ 0,  0,  0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0],
                            [ 1,  1,  1,  1,  1,  1,  1,  1],
                            [ 0,  0,  0,  0,  0,  0,  0,  0]])

        # KNIGHTS
        board[1] = np.array([[ 0, -1,  0,  0,  0,  0, -1,  0],
                             [ 0,  0,  0,  0,  0,  0,  0,  0],
                             [ 0,  0,  0,  0,  0,  0,  0,  0],
                             [ 0,  0,  0,  0,  0,  0,  0,  0],
                             [ 0,  0,  0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0],
                            [ 0,  1,  0,  0,  0,  0,  1,  0]])

        # BISHOPS
        board[2] = np.array([[ 0,  0, -1,  0,  0, -1,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0],
                            [ 0,  0,  1,  0,  0,  1,  0,  0]])

        # ROOK
        board[3] = np.array([[-1,  0,  0,  0,  0,  0,  0, -1],
                            [ 0,  0,  0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0],
                            [ 1,  0,  0,  0,  0,  0,  0,  1]])

        if self.p1_colour == "W":
            # KING
            board[4] = np.array([[ 0,  0,  0, -1,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  1,  0,  0,  0,  0]])

            # KING
            board[5] = np.array([[ 0,  0,  0,  0, -1,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  1,  0,  0,  0]])
        elif self.p1_colour == "B":
            # KING
            board[4] = np.array([[ 0,  0,  0,  0, -1,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  1,  0,  0,  0]])

            # KING
            board[5] = np.array([[ 0,  0,  0, -1,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  1,  0,  0,  0,  0]])
        else:
            raise Exception("Colour must be black or white")

        return board

    def __str__(self):
        out = ""

        trad_board = self.get_trad_board()

        for row_number, row in enumerate(trad_board):
            out += str(row_number + 1) + ")  "
            for col in row:
                out += col + " "
            out += "\n"

        out += "     A  B  C  D  E  F  G  H"
        out += "\n"

        return out

    # Takes a game state and returns a unique string key that contains the information
    # of the game.
    # This function "unrolls" the board into one dimension, and returns a string of length
    # 6 * 8 * 8 = 384 length string with 0 -> p2 piece, 1 -> empty, 2 -> p1 piece 
    def state_to_key(self, state) -> str:
        reshaped_state = np.reshape(state, newshape=[-1]) + 1
        out = ''.join(map(str, reshaped_state))
        return out

    # returns true if the game is over, and false if the game is not
    def game_ended(self, state) -> bool:
        # Checking to see if there a king is missing
        unique_king_values = set(np.unique(state[-1]).tolist())
        if unique_king_values != {-1, 0, 1}:
            return True

        # Checking to see if all other pieces are dead
        unique_other_piece_values = set(np.unique(state[:-1]).tolist())
        if unique_other_piece_values == {0}:
            return True

        return False

    def game_reward(self, state) -> float:
        assert self.game_ended(state) == True
        # Checking p1 and p2 kings are missing
        unique_king_values = set(np.unique(state[-1]).tolist())

        # P1 king:
        if 1 not in unique_king_values:
            return -1.0

        # P2 king:
        if -1 not in unique_king_values:
            return 1.0

        # If both kings are alive, and the game is done, that must mean it
        # is a tie

        return 0.0
        
    def q_index_to_chess_encoding(self, index):
        start_move_row = index // 512
        start_move_col = (index // 64) % 8

        end_move_row = (index // 8) % 8
        end_move_col = index % 8

        start = rc2str(start_move_col, start_move_row)
        end = rc2str(end_move_col, end_move_row)

        return (start, end)

    def chess_encoding_to_q_index(self, start, end):
        sm_row = ALPH_TO_INDEX[start[0]]
        sm_col = int(start[1]) - 1

        em_row = ALPH_TO_INDEX[end[0]]
        em_col = int(end[1]) - 1

        return (8**3 * sm_row) + (8**2 * sm_col) + (8 * em_row) + em_col

    def next_state(self, start_state : np.ndarray, move : tuple) -> np.ndarray:
        end_state = start_state.copy()

        rc1 = str2rc(move[0])
        rc2 = str2rc(move[1])

        end_state[:, rc2[0], rc2[1]] = end_state[:, rc1[0], rc1[1]]
        end_state[:, rc1[0], rc1[1]] = 0

        return end_state

    def print_state(self, state):
        out = ""

        trad_board = self.get_trad_board(state)

        for row_number, row in enumerate(trad_board):
            out += str(row_number + 1) + ")  "
            for col in row:
                out += col + " "
            out += "\n"

        out += "     A  B  C  D  E  F  G  H"
        out += "\n"

        return out

