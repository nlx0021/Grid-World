import random
import numpy as np
import matplotlib.pyplot as plt
from pylab import xticks, yticks
from tqdm import tqdm

from core.MDP import MDP
from utils.random_board import generate_random_board, generate_one_goal_board
from models.Grid_world import Grid_world


def run(H_range=[5,15],
        W_range=[5,15],
        gamma=.9, win_reward=1,
        step_size=1,
        mode="policy_iteration"):
    
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
            
            V_list =  one_grid_world.solve_mdp(mode=mode,
                                               init=False, verbose=True, need_return=True, step_size=step_size)
            
            step = len(V_list) - 1
            
            step_rec[H_idx, W_idx] = step
            
            Delta_theory = gamma ** (H+W-3) * (1-gamma)
            if mode == "policy_iteration":
                upper = np.log(2/(Delta_theory * (1-gamma))) / (1-gamma)
            elif mode == "projected_Q_descent":
                upper = 2 / (step_size*Delta_theory) * (1 / (step_size*(1-gamma)) + 1 / (1-gamma)**2) * (1/Delta_theory + step_size) - 1
            elif mode == "policy_descent":
                upper = 0
            
            upper_rec[H_idx, W_idx] = upper
            
            
    fig = plt.figure()
    ax3 = plt.axes(projection='3d')
    
    ax3.plot_surface(HH,WW,step_rec,cmap='rainbow') 
    # if mode == "policy_iteration":
    #     ax3.plot_surface(HH,WW,upper_rec,cmap='rainbow')
    
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



def fix_delta(L=40,
              gamma=.9, win_reward=1,
              step_size=1,
              mode="policy_iteration"):
    
    S_size_list = []
    step_list = []
    
    for H in tqdm(range(2, L-1)):
        
        W = L - H
        S_size = H * W
        
        board = generate_one_goal_board(H, W)
        
        one_grid_world = Grid_world(board,
                                    gamma,
                                    win_reward,
                                    punish_reward=-10)
        
        one_grid_world.mdp.init_policy_and_V(random_init=True)
        
        V_list =  one_grid_world.solve_mdp(mode=mode,
                                            init=False, verbose=True, need_return=True, step_size=step_size)
        
        print("Delta: %f" % one_grid_world.mdp.compute_delta())
        
        step = len(V_list) - 1  
        
        print(step)      
    
        S_size_list.append(S_size)
        step_list.append(step)
        
    _temp = list(zip(S_size_list, step_list))
    _temp = sorted(_temp, key=lambda x: x[0])
    
    S_size_list = [_[0] for _ in _temp]
    step_list = [_[1] for _ in _temp]
    
    plt.plot(S_size_list, step_list)
    plt.show()


if __name__ == '__main__':
    
    # run(mode="projected_Q_descent",
    #     step_size=100)
    
    # run(mode="policy_descent",
    #     step_size=100000)    
    
    fix_delta(L=50,
              step_size=10,
              mode="policy_iteration")
    
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