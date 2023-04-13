import numpy as np


def transform(player, game_state):
    game_state = np.array(game_state)
    game_state[game_state == 2] = -1

    state = np.zeros((2, game_state.shape[0], game_state.shape[1]), dtype=np.float32)

    if player == 2:
        game_state = game_state.T
        game_state *= -1

    for i in range(game_state.shape[0]):
        for j in range(game_state.shape[1]):
            if game_state[i][j] == 1:
                state[0][i][j] = 1
            elif game_state[i][j] == -1:
                state[1][i][j] = 1

    return state.tolist()