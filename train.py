from vindchess.game import GameManager, Game

def main():
    game_manager = GameManager()
    print(game_manager.game)
    print(game_manager.game.get_available_moves(player = 1))
    print(len(game_manager.game.get_available_moves(player = 1)))

if __name__ == "__main__":
    main()