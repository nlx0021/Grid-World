import random
import numpy as np
import matplotlib.pyplot as plt
from pylab import xticks, yticks

from core.MDP import MDP
from utils.utils import phi_exp_factory, phi_exp_inv_factory, phi_exp_combined_factory

class TrivialMDP():
    
    def __init__(self,
                 S_size=70,
                 A_size=10,
                 gamma=.9,
                 seed=21,
                 entropy_coeff=None,
                 reward_value=0.0,
                 transition_mode="self_loop"):
        
        """
        Generate a trivial MDP where all actions are optimal.

        Key idea:
            For every state s and every action a,
            all actions have identical transition probabilities
            and identical rewards. Therefore Q(s, a) is the same
            for all actions a.

        Parameters:
            S_size:
                Number of states.
            A_size:
                Number of actions.
            gamma:
                Discount factor.
            seed:
                Random seed. Only used if transition_mode == "random_same".
            entropy_coeff:
                Entropy regularization coefficient.
            reward_value:
                Constant reward for every (s, a, s').
            transition_mode:
                "self_loop": each state transitions to itself.
                "uniform": each action transitions uniformly to all states.
                "random_same": all actions share the same random transition matrix.
        """
        
        self.S_size = S_size
        self.A_size = A_size
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        
        assert transition_mode in ["self_loop", "uniform", "random_same"]
        
        # ----------------------------------------------------
        # 1. Construct transition matrix P[a, s, s']
        # ----------------------------------------------------
        
        self.P = np.zeros((A_size, S_size, S_size))
        
        if transition_mode == "self_loop":
            base_P = np.eye(S_size)
            
        elif transition_mode == "uniform":
            base_P = np.ones((S_size, S_size)) / S_size
            
        elif transition_mode == "random_same":
            np.random.seed(seed)
            base_P = np.random.uniform(size=(S_size, S_size), low=0, high=1)
            base_P = base_P / base_P.sum(axis=1, keepdims=True)
        
        # All actions share the same transition matrix.
        for a in range(A_size):
            self.P[a] = base_P.copy()
        
        # ----------------------------------------------------
        # 2. Construct reward tensor r[a, s, s']
        # ----------------------------------------------------
        # Same reward for every action, state, and next state.
        # Hence every action has exactly the same Q-value.
        
        self.rewards = np.full(
            shape=(A_size, S_size, S_size),
            fill_value=reward_value,
            dtype=float
        )
        
        # ----------------------------------------------------
        # 3. Build MDP object
        # ----------------------------------------------------
        
        self.mdp = MDP(
            self.P,
            self.gamma,
            self.rewards,
            entropy_coeff=entropy_coeff
        )
        
        
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
        
        """
        Solve the trivial MDP using the same interface as RandomMDP.
        """
        
        assert mode["alg"] in ["value_iteration", "policy_iteration",
                               "projected_Q_descent", "policy_descent",
                               "softmax", "softmax_adaptive", "softmax_temp", "softmax_NPG",
                               "phi", "escort", "escort_normalized", "mirror_descent"]
        
        alg = mode["alg"]
        
        if init:
            self.mdp.init_policy_and_V(random_init=True)
        
        if not verbose:
            print("Solving!")
        
        if alg == "value_iteration":
            return_dict = self.mdp.value_iteration(
                epsilon=epsilon,
                max_iter=max_iter,
                asynchronous=asynchronous,
                need_return=need_return,
                silence=verbose,
                seed=seed
            )
            
        elif alg == "policy_iteration":
            return_dict = self.mdp.policy_iteration(
                max_iter=max_iter,
                need_return=need_return,
                silence=verbose,
                seed=seed
            )
            
        elif alg == "projected_Q_descent":
            return_dict = self.mdp.projected_Q_descent(
                max_iter=max_iter,
                step_size=step_size,
                need_return=need_return,
                silence=verbose,
                noise=noise,
                seed=seed,
                step_size_increasing=step_size_increasing
            )
            
        elif alg == "policy_descent":
            return_dict = self.mdp.projected_Q_descent(
                max_iter=max_iter,
                step_size=step_size,
                need_return=need_return,
                silence=verbose,
                mode="policy_descent",
                noise=noise,
                seed=seed,
                step_size_increasing=step_size_increasing
            )
            
        elif alg == "softmax":
            return_dict = self.mdp.softmax_descent(
                max_iter=max_iter,
                step_size=step_size,
                need_return=need_return,
                silence=verbose,
                noise=noise,
                seed=seed
            )
            
        elif alg == "softmax_adaptive":
            return_dict = self.mdp.softmax_descent(
                max_iter=max_iter,
                step_size=step_size,
                need_return=need_return,
                silence=verbose,
                mode="adaptive",
                noise=noise,
                seed=seed
            )
            
        elif alg == "softmax_NPG":
            return_dict = self.mdp.softmax_descent(
                max_iter=max_iter,
                step_size=step_size,
                need_return=need_return,
                silence=verbose,
                mode="NPG",
                noise=noise,
                seed=seed
            )
            
        elif alg == "escort_normalized":
            p = mode["p"]
            return_dict = self.mdp.escort_descent(
                max_iter=max_iter,
                step_size=step_size,
                need_return=need_return,
                silence=verbose,
                mode="normalized",
                p=p,
                noise=noise,
                seed=seed
            )
            
        elif alg == "escort":
            p = mode["p"]
            return_dict = self.mdp.escort_descent(
                max_iter=max_iter,
                step_size=step_size,
                need_return=need_return,
                silence=verbose,
                mode="origin",
                p=p,
                noise=noise,
                seed=seed
            )
            
        elif alg == "phi": 
            phi = mode["phi"]
            step_include_d = mode.get("step_include_d", False)
            return_dict = self.mdp.phi_policy_update(
                phi,
                max_iter=max_iter,
                step_size=step_size,
                need_return=need_return,
                silence=verbose,
                noise=noise,
                step_include_d=step_include_d,
                seed=seed
            )
            
        elif alg == "mirror_descent":
            mirror_funcs = mode["mirror_funcs"]
            return_dict = self.mdp.mirror_descent(
                mirror_funcs,
                max_iter=max_iter,
                step_size=step_size,
                need_return=need_return,
                silence=verbose,
                noise=noise,
                seed=seed
            )
        
        if need_return:
            return return_dict
