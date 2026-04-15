import random
import numpy as np
import matplotlib.pyplot as plt
from pylab import xticks, yticks

from core.MDP import MDP
from utils.utils import phi_exp_factory, phi_exp_inv_factory, phi_exp_combined_factory


class RandomMDP():
    
    def __init__(self,
                 S_size=70,
                 A_size=10,
                 gamma=.9,
                 seed=21,
                 entropy_coeff=None):
        
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
        self.rewards = np.expand_dims(self.rewards, axis=2).repeat(S_size, 2)
        
        self.mdp = MDP(self.P, self.gamma, self.rewards, entropy_coeff=entropy_coeff)
        
        
    def solve_mdp(self,
                  mode={"alg": "policy_iteration"},
                  max_iter=10000,
                  epsilon=1e-7,
                  step_size=1,
                  asynchronous=False,
                  init=False,
                  verbose=False,
                  need_return=False,
                  noise=None,
                  seed=21,
                  step_size_increasing=False):
        '''
        求解Grid world问题。
        参数：
            mode: 
                {"alg": "policy_iteration", "label": "policy_iteration"}
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
        
        assert mode["alg"] in ["value_iteration", "policy_iteration",
                               "projected_Q_descent", "policy_descent",
                               "softmax", "softmax_adaptive", "softmax_temp", "softmax_NPG",
                               "phi", "escort", "escort_normalized"]
        
        alg = mode["alg"]
        
        if init:
            self.mdp.init_policy_and_V(random_init=True)
        
        if not verbose:
            print("Solving!")
        
        if alg == "value_iteration":
            return_dict = self.mdp.value_iteration(epsilon=epsilon, max_iter=max_iter, asynchronous=asynchronous,
                                              need_return=need_return, silence=verbose, seed=seed)
        elif alg == "policy_iteration":
            return_dict = self.mdp.policy_iteration(max_iter=max_iter,
                                              need_return=need_return, silence=verbose, seed=seed)
        elif alg == "projected_Q_descent":
            return_dict = self.mdp.projected_Q_descent(max_iter=max_iter, step_size=step_size,
                                                  need_return=need_return, silence=verbose, noise=noise, seed=seed, step_size_increasing=step_size_increasing)
        elif alg == "policy_descent":
            return_dict = self.mdp.projected_Q_descent(max_iter=max_iter, step_size=step_size,
                                                  need_return=need_return, silence=verbose, mode="policy_descent", noise=noise, seed=seed, step_size_increasing=step_size_increasing)
        elif alg == "softmax":
            return_dict = self.mdp.softmax_descent(max_iter=max_iter, step_size=step_size,
                                              need_return=need_return, silence=verbose, noise=noise, seed=seed)
        elif alg == "softmax_adaptive":
            return_dict = self.mdp.softmax_descent(max_iter=max_iter, step_size=step_size,
                                              need_return=need_return, silence=verbose, mode="adaptive", noise=noise, seed=seed)     
        # elif alg == "softmax_temp":
        #     return_dict = self.mdp.softmax_descent(max_iter=max_iter, step_size=step_size,
        #                                       need_return=need_return, silence=verbose, mode="temp", noise=noise, seed=seed)    
        elif alg == "softmax_NPG":
            return_dict = self.mdp.softmax_descent(max_iter=max_iter, step_size=step_size,
                                              need_return=need_return, silence=verbose, mode="NPG", noise=noise, seed=seed)  
        elif alg == "escort_normalized":
            p = mode["p"]
            return_dict = self.mdp.escort_descent(max_iter=max_iter, step_size=step_size,
                                                  need_return=need_return, silence=verbose, mode="normalized", p=p, noise=noise, seed=seed)
        elif alg == "escort":
            p = mode["p"]
            return_dict = self.mdp.escort_descent(max_iter=max_iter, step_size=step_size,
                                                  need_return=need_return, silence=verbose, mode="origin", p=p, noise=noise, seed=seed)                   
        elif alg == "phi": 
            phi = mode["phi"]
            step_include_d = mode.get("step_include_d", False)
            return_dict = self.mdp.phi_policy_update(phi, max_iter=max_iter, step_size=step_size,
                                                     need_return=need_return, silence=verbose, noise=noise, step_include_d=step_include_d, seed=seed)
        if need_return:
            return return_dict    
        
    def TD_PMD(self,
                seed=21,
                init_state=0,
                init_action=0,
                samples_N=10000,
                actor_step_size_func=lambda k: 0.1,
                critic_step_size_func=lambda k: 0.1,
                eps_func=lambda k: 0.1,
                batch_size=1,
                mirror_map="npg",
                use_eps=True,
                adaptive_critic_step_size=False,
                adaptive_actor_step_size=False,
                off_policy=False,
                vartheta=1,
                is_expected=False,
                is_approximated=False):
    
        return_dict = self.mdp.single_loop_actor_to_critic_PMD(
            seed=seed,
            init_state=init_state,
            init_action=init_action,
            samples_N=samples_N,
            actor_step_size_func=actor_step_size_func,
            critic_step_size_func=critic_step_size_func,
            batch_size=batch_size,
            mirror_map=mirror_map,
            eps_func=eps_func,
            use_eps=use_eps,
            adaptive_critic_step_size=adaptive_critic_step_size,
            adaptive_actor_step_size=adaptive_actor_step_size,
            off_policy=off_policy,
            vartheta=vartheta,
            is_expected=is_expected,
            is_approximated=is_approximated
        )
        
        return return_dict
    
    
    def V_curve_in_VI(self,
                      raw_V_list):
        
        '''
        将VI返回的V_list中，每个V值所对应的贪心策略的真实V值给计算出来
        '''
        
        V_policy_list = []
        
        for V in raw_V_list:
            self.mdp.set_V(V)
            self.mdp.extract_policy()
            self.mdp.evaluate_policy()
            V_policy_list.append(self.mdp.V.copy())
            
        return V_policy_list
    
    
    def TD_policy_evaluation(self,
                             max_iter=1000,
                             epsilon=None,
                             max_length=10000,
                             verbose=False,
                             fix_steps_size=1.,
                             seed=21,
                             step_size_scale=True):
        
        # target_state_idx_list = [self.pos2idx[target_state] for target_state in self.target_state_list]
        V_gap_list = self.mdp.TD_policy_evaluation(
            max_iter=max_iter,
            epsilon=epsilon,
            max_length=max_length,
            terminate_state=None,
            need_return=True,
            fix_step_size=fix_steps_size,
            seed=seed,
            step_size_scale=step_size_scale
        )
        
        V_gap_array = np.stack(V_gap_list, axis=0)
        V_gap_mean_curse = np.mean(np.abs(V_gap_array), axis=1).tolist()
        if not verbose:
            plt.plot(V_gap_mean_curse)
            plt.show()
        plt.clf()
            