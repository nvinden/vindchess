from .model import load_policy_net

import random
import numpy as np

from itertools import product

PIECE_TO_INDEX = {"p": 0, "k": 1, "b": 2, "r": 3, "Q": 4, "K": 5}
INDEX_TO_PIECE = {val: key for key, val in PIECE_TO_INDEX.items()}

GET_OPPONENT_COLOUR = {"W": "B", "B": "W"}

ALL_STRATEGIES = ["vindgod", "random", "simple_heuristic"]

def rc2str(row, column):
    column_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
    return column_letters[column] + str(row + 1)

def on_board(row, column):
    return row >= 0 and row <= 7 and column >= 0 and column <= 7

class GameManager():
    def __init__(self, n_games = 1):
        policy_net = load_policy_net()

        self.game = Game()

    def play_game(self, p1_strategy : str, p2_strategy):
        assert p1_strategy in ALL_STRATEGIES
        assert p2_strategy in ALL_STRATEGIES

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
    def get_available_moves(self, player) -> list:
        assert player in [1, 2]

        if player == 1:
            return self.__get_available_p1_moves()
        elif player == 2:
            return self.__get_available_p2_moves()

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
        
    
    def __get_available_p1_moves(self):
        board = self.board.copy()

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

    def get_trad_board(self):
        board = self.board.copy()
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

        return out