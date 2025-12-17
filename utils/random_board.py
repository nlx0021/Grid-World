import numpy as np

def generate_random_board(H, W, p_1, p_2, seed=21):
    
    def random_idx(p):
        prob = np.random.uniform(size=board.shape)
        return np.where(prob < p)
    np.random.seed(seed)
    board = np.zeros((H, W), dtype=np.uint8)
    board[random_idx(p_1)] = 1
    board[random_idx(p_2)] = 2
    
    # import pdb; pdb.set_trace()
    
    return board


def generate_one_goal_board(H, W, random=False, seed=21):
    
    board = np.zeros((H, W), dtype=np.uint8)
    # board[np.random.choice(H), np.random.choice(W)] = 2
    if not random:
        board[0, 0] = 2
    else:
        np.random.seed(seed)
        rnd_H_idx = np.random.randint(H)
        rnd_W_idx = np.random.randint(W)
        board[rnd_H_idx, rnd_W_idx] = 2
    
    return board