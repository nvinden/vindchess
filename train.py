from vindchess.game import GameManager, Game

def main():
    game_manager = GameManager()
    print(game_manager.game)
    
    game_manager.play_game("vindgod", "random")

if __name__ == "__main__":
    main()