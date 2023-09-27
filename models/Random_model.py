import random
import numpy as np
import matplotlib.pyplot as plt
from pylab import xticks, yticks

from core.MDP import MDP


class RandomMDP():
    
    def __init__(self,
                 S_size=70,
                 A_size=10,
                 gamma=.9,
                 seed=21):
        
        '''
        Randomly generate a MDP model.
        '''
        
        self.S_size = S_size
        self.A_size = A_size
        self.gamma = gamma
        
        # Randomly generate MDP.
        np.random.seed(seed)
        self.P = np.random.uniform(size=(A_size, S_size, S_size), low=0, high=1)
        for a in range(A_size): 
            for s in range(S_size):
                self.P[a,s] = self.P[a,s] / np.sum(self.P[a,s])    # Normalize.
        self.rewards = np.random.uniform(size=(A_size, S_size))
        self.rewards = np.expand_dims(self.rewards, axis=2).repeat(S_size)
        
        self.mdp = MDP(self.P, self.gamma, self.rewards)
        
        
    def solve_mdp(self,
                  mode="value_iteration",
                  max_iter=10000,
                  epsilon=1e-7,
                  step_size=1,
                  asynchronous=False,
                  init=False,
                  verbose=False,
                  need_return=False):
        '''
        求解Grid world问题。
        参数：
            mode: 
                "value_iteration" or "policy_iteration".
            max_iter:
                最大迭代数。
            epsilon:
                终止误差阈值。
            step_size (只对Q-descent类算法有效):
                步长。
            asynchoronous:
                是否采用异步更新 (仅针对value iteration)
            init:
                是否在求解前将MDP的V和policy进行初始化？
            verbose:
                是否进行plt的show输出？
            need_return:
                是否返回V_list？
        '''
        
        assert mode in ["value_iteration", "policy_iteration", "projected_Q_descent", "policy_descent"]
        
        if init:
            self.mdp.init_policy_and_V(random_init=True)
        
        if not verbose:
            print("Solving!")
        
        if mode == "value_iteration":
            V_list = self.mdp.value_iteration(epsilon=epsilon, max_iter=max_iter, asynchronous=asynchronous,
                                              need_return=need_return, silence=verbose)
        elif mode == "policy_iteration":
            V_list = self.mdp.policy_iteration(max_iter=max_iter,
                                              need_return=need_return, silence=verbose)
        elif mode == "projected_Q_descent":
            V_list = self.mdp.projected_Q_descent(max_iter=max_iter, step_size=step_size,
                                                  need_return=need_return, silence=verbose)
        elif mode == "policy_descent":
            V_list = self.mdp.projected_Q_descent(max_iter=max_iter, step_size=step_size,
                                                  need_return=need_return, silence=verbose, mode="policy_descent")
        
        if not verbose:
            self.visualize_policy()
            
        if need_return:
            return V_list    