CHESSNET_CONFIG = {
    "n_convolutions_per_block": 3,
    "convolutional_filter_depth": [6, 16, 32, 64], 
}

TRAIN_CONFIG = {
    "number_of_iters": 10,
    "number_of_episodes": 10,
    "number_MCTS_sims": 100,
    "c_puct": 1.0,
    "number_of_epochs": 10,
    "lr": 0.001,
    "weight_decay": 1e-5,
    "batch_size": 8
}