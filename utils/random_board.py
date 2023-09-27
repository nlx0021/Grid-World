import numpy as np

def generate_random_board(H, W, p_1, p_2):
    
    def random_idx(p):
        prob = np.random.uniform(size=board.shape)
        return np.where(prob < p)
    
    board = np.zeros((H, W), dtype=np.uint8)
    board[random_idx(p_1)] = 1
    board[random_idx(p_2)] = 2
    
    # import pdb; pdb.set_trace()
    
    return board


def generate_one_goal_board(H, W):
    
    board = np.zeros((H, W), dtype=np.uint8)
    # board[np.random.choice(H), np.random.choice(W)] = 2
    board[0, 0] = 2
    
    return board