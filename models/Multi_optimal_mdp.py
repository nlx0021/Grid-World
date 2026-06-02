import random
import numpy as np
import matplotlib.pyplot as plt
from pylab import xticks, yticks

from core.MDP import MDP
from utils.utils import phi_exp_factory, phi_exp_inv_factory, phi_exp_combined_factory

class MultiOptimalMDP:
    def __init__(
        self,
        S_size=70,
        A_size=10,
        gamma=0.9,
        seed=21,
        num_opt_actions=2,
        gap_min=0.01,
        gap_max=0.03,
        value_center=None,
        value_variation=0.10,
        entropy_coeff=None,
        check_reward_range=True
    ):
        """
        Generate a more complex MDP where every state has multiple optimal actions.

        Construction idea:
        1. Randomly generate transition kernels P[a, s, s'] for every action.
        2. Choose a target optimal value function V_star.
        3. Define rewards so that

               Q_star(s,a) = V_star(s) - gap(s,a).

           For optimal actions, gap(s,a)=0.
           For suboptimal actions, gap(s,a)>0.

        Therefore, for every state s,

               A_s^* = {0, 1, ..., num_opt_actions - 1}.

        Unlike the simple version, different actions have different transition kernels.
        """

        assert A_size >= 3, "Need at least 3 actions: at least 2 optimal and 1 suboptimal."
        assert 2 <= num_opt_actions < A_size, "Need 2 <= num_opt_actions < A_size."
        assert 0 < gap_min <= gap_max, "Need 0 < gap_min <= gap_max."
        assert 0 <= gamma < 1, "Need gamma in [0,1)."

        self.S_size = S_size
        self.A_size = A_size
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.num_opt_actions = num_opt_actions

        rng = np.random.default_rng(seed)

        # -------------------------------------------------
        # 1. Random transition kernels for every action
        # -------------------------------------------------
        self.P = rng.uniform(low=0.0, high=1.0, size=(A_size, S_size, S_size))
        self.P = self.P / self.P.sum(axis=2, keepdims=True)

        # -------------------------------------------------
        # 2. Choose a target V_star
        # -------------------------------------------------
        # If rewards are roughly in [0,1], a natural value scale is around 1/(2(1-gamma)).
        if value_center is None:
            value_center = 0.5 / (1.0 - gamma)

        raw = rng.normal(size=S_size)
        raw = raw - raw.mean()
        raw = raw / (np.max(np.abs(raw)) + 1e-12)

        self.V_star_target = value_center + value_variation * raw

        # -------------------------------------------------
        # 3. Define action gaps
        # -------------------------------------------------
        # First num_opt_actions are optimal at every state.
        gaps = rng.uniform(low=gap_min, high=gap_max, size=(A_size, S_size))
        gaps[:num_opt_actions, :] = 0.0
        self.gaps = gaps

        # -------------------------------------------------
        # 4. Back out rewards from Bellman consistency
        # -------------------------------------------------
        # Need:
        #     Q_star(s,a) = r(s,a) + gamma P[a,s,:] @ V_star
        #                  = V_star(s) - gap(a,s)
        #
        # Thus:
        #     r(s,a) = V_star(s) - gap(a,s) - gamma P[a,s,:] @ V_star
        rewards_by_action_state = np.zeros((A_size, S_size), dtype=float)

        for a in range(A_size):
            rewards_by_action_state[a] = (
                self.V_star_target
                - gaps[a]
                - gamma * self.P[a].dot(self.V_star_target)
            )

        self.rewards_by_action_state = rewards_by_action_state

        if check_reward_range:
            r_min = rewards_by_action_state.min()
            r_max = rewards_by_action_state.max()

            if r_min < 0.0 or r_max > 1.0:
                raise ValueError(
                    f"Generated rewards are outside [0,1]: min={r_min:.4f}, max={r_max:.4f}. "
                    "Try smaller value_variation, smaller gap_max, or set check_reward_range=False."
                )

        # Match existing MDP format: rewards shape = (A, S, S)
        self.rewards = np.expand_dims(rewards_by_action_state, axis=2).repeat(S_size, axis=2)

        # True optimal action sets
        self.optimal_action_sets = {
            s: list(range(num_opt_actions))
            for s in range(S_size)
        }

        # Build MDP object
        self.mdp = MDP(
            self.P,
            self.gamma,
            self.rewards,
            entropy_coeff=entropy_coeff
        )

    def verify_construction(self, tol=1e-10):
        """
        Verify the designed Q_star relation:
            Q_star(s,a) = V_star(s) - gap(a,s).
        """
        Q_star = np.zeros((self.S_size, self.A_size))

        for s in range(self.S_size):
            for a in range(self.A_size):
                Q_star[s, a] = (
                    self.rewards_by_action_state[a, s]
                    + self.gamma * self.P[a, s].dot(self.V_star_target)
                )

        target_Q = np.zeros_like(Q_star)
        for s in range(self.S_size):
            for a in range(self.A_size):
                target_Q[s, a] = self.V_star_target[s] - self.gaps[a, s]

        max_error = np.max(np.abs(Q_star - target_Q))

        opt_sets = []
        for s in range(self.S_size):
            max_q = np.max(Q_star[s])
            opt_actions = np.where(np.abs(Q_star[s] - max_q) <= tol)[0].tolist()
            opt_sets.append(opt_actions)

        return {
            "max_bellman_consistency_error": max_error,
            "optimal_action_sets": opt_sets,
            "reward_min": self.rewards_by_action_state.min(),
            "reward_max": self.rewards_by_action_state.max(),
            "V_star_min": self.V_star_target.min(),
            "V_star_max": self.V_star_target.max(),
        }
        
    def solve_mdp(self,
                  mode="value_iteration",
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
        elif alg == "softmax_temp":
            return_dict = self.mdp.softmax_descent(max_iter=max_iter, step_size=step_size,
                                              need_return=need_return, silence=verbose, mode="temp", noise=noise, seed=seed)    
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
            return_dict = self.mdp.phi_policy_update(phi, max_iter=max_iter, step_size=step_size,
                                                     need_return=need_return, silence=verbose, noise=noise, seed=seed)
        elif alg == "mirror_descent":
            mirror_funcs = mode["mirror_funcs"]
            return_dict = self.mdp.mirror_descent(mirror_funcs, max_iter=max_iter, step_size=step_size,
                                                  need_return=need_return, silence=verbose, noise=noise, seed=seed)
                        
        if need_return:
            return return_dict      
    