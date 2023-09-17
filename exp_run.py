import random
import numpy as np
import matplotlib.pyplot as plt
from pylab import xticks, yticks
from tqdm import tqdm

from MDP import MDP
from random_board import generate_random_board, generate_one_goal_board
from Grid_world import Grid_world


def run(H_range=[5,30],
        W_range=[5,30],
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
            
            V_list =  one_grid_world.solve_mdp(mode="policy_iteration",
                                               init=True, verbose=True, need_return=True)
            
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
            
          
            






if __name__ == '__main__':
    
    run()