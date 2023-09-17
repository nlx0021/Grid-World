import random
import numpy as np
import matplotlib.pyplot as plt
from pylab import xticks, yticks
from tqdm import tqdm

from MDP import MDP
from random_board import generate_random_board, generate_one_goal_board
from Grid_world import Grid_world


def run(H_range=[5,15],
        W_range=[5,15],
        gamma=.9, win_reward=1):
    
    hh = np.arange(H_range[0], H_range[1]+1)
    ww = np.arange(W_range[0], W_range[1]+1)
    HH, WW = np.meshgrid(hh, ww)
    
    step_rec = np.zeros_like(HH)
    upper_rec = np.zeros_like(HH)
    
    for H_idx, H in tqdm(enumerate(range(H_range[0], H_range[1]+1))):
        for W_idx, W in enumerate(range(W_range[0], W_range[1]+1)):
            
            board = generate_one_goal_board(H, W)
            
            one_grid_world = Grid_world(board,
                                        gamma,
                                        win_reward,
                                        punish_reward=-10)
            
            one_grid_world.mdp.init_policy_and_V(random_init=True)
            
            V_list =  one_grid_world.solve_mdp(mode="policy_iteration",
                                               init=False, verbose=True, need_return=True)
            
            step = len(V_list) - 1
            
            step_rec[H_idx, W_idx] = step
            
            Delta_theory = gamma ** (H+W-3) * (1-gamma)
            upper = np.log(2/(Delta_theory * (1-gamma))) / (1-gamma)
            
            upper_rec[H_idx, W_idx] = upper
            
            
    fig = plt.figure()
    ax3 = plt.axes(projection='3d')
    
    ax3.plot_surface(HH,WW,step_rec,cmap='rainbow') 
    ax3.plot_surface(HH,WW,upper_rec,cmap='rainbow')
    
    plt.show()
            
          
            
def vis_process(grid_world: Grid_world):
    
    V_old = -100 * np.ones((grid_world.H * grid_world.W))  
    grid_world.mdp.init_policy_and_V(random_init=True)  
    grid_world.visualize_policy()
    while True:
        
        V_list = grid_world.solve_mdp(
            mode="policy_iteration",
            max_iter=1,
            need_return=True
        )
        
        # grid_world.visualize_policy()

        V = V_list[1]
        if np.linalg.norm(V_old - V, ord=np.inf) < 1e-23:
            
            break
        
        V_old = V
        print(np.linalg.norm(V))




if __name__ == '__main__':
    
    run()
    
    # random.seed(21)
    # H = 3; W = 25
    # # board = generate_random_board(10, 10, .2, .05)
    # board = generate_one_goal_board(H, W)
    
    # gamma = .9
    # win_reward = 1,
    # punish_reward = -10
    
    # one_grid_world = Grid_world(board,
    #                             gamma,
    #                             win_reward,
    #                             punish_reward)   
    
    # vis_process(one_grid_world) 
    # V_list = one_grid_world.solve_mdp(mode="policy_iteration",
    #                                   need_return=True)
    # [print(np.linalg.norm(V, ord=1)) for V in V_list]