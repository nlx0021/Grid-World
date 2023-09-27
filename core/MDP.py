import numpy as np
import random
from tqdm import tqdm

from utils.utils import compute_visit_prob

class MDP():
    
    def __init__(self,
                 P,
                 gamma,
                 rewards):
        '''
        初始化一个MDP模型。
        参数：
            P: 
                状态转移模型。
                其为一个[A, N, N]的数组，A和N分别代表动作空间和状态空间大小。
                a，i，j位置表示采取a动作后从i状态转移到j状态的概率。
            gamma:
                折扣因子。
            rewards:
                即时奖励。
                也为一个[A, N, N]的数组。a，i，j位置表示(i, a, j)的即时奖励。
        注意：
            状态空间S和动作空间A按照默认的顺序标号方式表示，已暗含在参数P和rewards中。
        '''
        
        assert P.shape == rewards.shape, "P和rewards的形状应该一致"
        self.P = P
        self.gamma = gamma 
        self.rewards = rewards
        
        self.A_size, self.S_size, _ = self.P.shape
        assert self.A_size > 1 and self.S_size > 1, "动作空间和状态空间不能大小为1"
        
        self.init_policy_and_V()
        
        
    def value_iteration(self,
                        epsilon=1e-5,
                        max_iter=1000,
                        asynchronous=False,
                        need_return=False,
                        silence=False):
        '''
        值迭代算法计算最佳效用值，并提取最佳策略。
        参数：
            epsilon: 
                迭代的误差阈值。
            max_iter:
                迭代次数的上限。
            asynchronous:
                是否采取异步更新？（异步更新将使用顺序更新）
            need_return:
                是否返回VI中每一步的V值？
            silence:
                是否进行输出？
        '''
        
        V = self.V.copy()
        P = self.P
        rewards = self.rewards
        gamma = self.gamma
        
        iter = 0
        V_list = [V.copy()]
        while True:
            iter += 1
            # Update.
            V_new = V.copy()
            
            ''' Element-wise form. '''
            
            if asynchronous:
                for s in range(self.S_size):
                    temp = []
                    for a in range(self.A_size):   # Compute the new V-value.
                        q = np.dot(P[a, s, :], rewards[a, s, :] + gamma * V_new)
                        # q = np.dot(P[a, s, :], rewards[a, s, :] + gamma * V)    For synchronous element-wise form.
                        temp.append(q)
                    V_new[s] = max(temp)
            
            else:
                ### Vector form. (Only support synchronous update) ###
                V_new = np.diag(np.max(np.einsum('ijk,ikl->ijl', P, np.transpose(rewards, [0,2,1]) + gamma*V.reshape((-1,1))), axis=0))     
            V_list.append(V_new.copy())
            
            if np.linalg.norm(V - V_new) < epsilon:
                self.V = V_new.copy()
                if not silence:
                    print("值迭代收敛，迭代次数为: %d" % iter)
                break
            
            elif iter >= max_iter:
                self.V = V_new.copy()
                if not silence:
                    print("值迭代未收敛！")
                return 
            
            V = V_new.copy()
        
        self.V = V
        self.extract_policy()
        
        return V_list
                
                
    def policy_iteration(self,
                         max_iter=100,
                         need_return=False,
                         silence=False):
        '''
        策略迭代算法计算最佳策略。
        参数：
            max_iter:
                迭代次数的上限。
            need_return:
                是否需要返回V_list？
            silence:
                是否不输出任何信息？
        '''        
        
        iter = 0
        V_list = [self.V.copy()]
        while True:
            iter += 1
            V_old = self.V.copy()
            self.evaluate_policy()
            V_new = self.V.copy()
            V_list.append(V_new)
            
            if np.linalg.norm(V_new - V_old, ord=np.inf) < 1e-13:
                if not silence:
                    print("策略迭代收敛，迭代次数为：%d" % iter)
                break                
            
            # policy_old = self.policy.copy()
            self.extract_policy()
            
            # if policy_old == self.policy:
            #     print("策略迭代收敛，迭代次数为：%d" % iter)
            #     break
                
            if iter >= max_iter:
                if not silence:
                    print("策略迭代未收敛！")
                break
            
        return V_list
    
    
    def projected_Q_descent(self,
                            max_iter=1000,
                            step_size=1,
                            need_return=False,
                            silence=False,
                            mode="projected_Q_descent"):
        '''
        Projected Q-descent算法。
        参数：
            max_iter:
                迭代次数的上限。
            step_size:
                步长值。
            need_return:
                是否需要返回V_list？
            silence:
                是否不输出任何信息？    
        '''
        
        def proj_to_simplex(prob_policy):
            
            for s in range(self.S_size):
                _policy_s = prob_policy[s]
                _policy_s_sorted = np.sort(_policy_s)
                
                for i in range(self.A_size-1, 0, -1):
                    t_i = (np.sum(_policy_s_sorted[i:]) - 1) / (self.A_size - i)
                    if t_i >= _policy_s_sorted[i-1]:
                        t = t_i
                        break
                
                else:
                    t = (np.sum(_policy_s_sorted) - 1) / self.A_size
                
                prob_policy[s] = np.clip(_policy_s - t, a_min=0, a_max=1)
                prob_policy[s] = prob_policy[s] / np.sum(prob_policy[s])

            return prob_policy
                
                
        iter = 0
        V_list = []
        
        # First we run a baseline.
        self.policy_iteration(max_iter=1000,
                              silence=True)
        
        V_star = self.V.copy()
        
        self.init_policy_and_V(random_init=True)
        
        while True:
            iter += 1
            self.evaluate_policy(use_prob_policy=True)
            V_new = self.V.copy()
            V_list.append(V_new)
            
            if np.linalg.norm(V_new - V_star, ord=np.inf) < 1e-13:
                if not silence:
                    print("Q下降算法收敛，迭代次数为：%d" % iter)
                break             
            
            if mode == "projected_Q_descent":
                self.prob_policy = proj_to_simplex(self.prob_policy + step_size * self.Q)
            elif mode == "policy_descent":
                # If we use policy gradient, then we need to compute visit prob.
                d = compute_visit_prob(self.P,
                                       self.prob_policy,
                                       init_dist=np.ones((self.S_size, )) / self.S_size,
                                       gamma=self.gamma)   
                self.prob_policy = proj_to_simplex(self.prob_policy + step_size / (1-self.gamma) * d.reshape((-1,1)) * self.Q)             
            
            if iter >= max_iter:
                if not silence:
                    print("策略迭代未收敛！")
                break
            
        return V_list            
        
    
    def evaluate_policy(self, epsilon=None,
                        use_prob_policy=False):
        '''
        从策略中计算出对应的效用值。
        若epsilon指定，则对epsilon-greedy policy进行评估
        若use_prob_policy指定，则对概率策略进行评估
        '''
        
        assert not ((epsilon is not None) and use_prob_policy)
        
        P = self.P
        rewards = self.rewards
        gamma = self.gamma
        policy = self.policy  
        prob_policy = self.prob_policy      
        
        # V_pi = np.linalg.inv(I - gamma * P_pi) @ r_pi
        r_all = np.sum(P * rewards, axis=2)
        r_pi = np.zeros_like(r_all[0])      # Size: [S_size]
        for s in range(self.S_size):
            if epsilon is None:
                if not use_prob_policy:
                    r_pi[s] = r_all[policy[s], s]
                else:
                    r_pi[s] = np.dot(r_all[:, s], prob_policy[s, :])
            else:
                r_pi[s] = (1-epsilon) * r_all[policy[s], s] + epsilon/self.A_size * np.sum(r_all[:, s], axis=0)
            
        P_pi = np.zeros_like(P[0])          # Size: [S_size, S_size]
        for s in range(self.S_size):
            if epsilon is None:
                if not use_prob_policy:
                    P_pi[s, :] = P[policy[s], s, :]
                else:
                    P_pi[s, :] = np.dot(prob_policy[s, :], P[:, s, :]).reshape((-1,))
            else:
                P_pi[s, :] = (1-epsilon) * P[policy[s], s, :] + epsilon/self.A_size * np.sum(P[:, s, :], axis=0)
        
        V_pi = np.linalg.inv(np.eye(self.S_size) - gamma * P_pi) @ r_pi.reshape((-1,1))
        self.V = V_pi.reshape((-1,))
        
        Q_pi = np.stack([np.diag(_) for _ in np.einsum('ijk,ikl->ijl', P, np.transpose(rewards, [0,2,1]) + gamma*self.V.reshape((-1,1)))])
        self.Q = Q_pi.transpose()
        
        # We need to synchronize the policy and prob_policy.
        if use_prob_policy:
            for s in range(self.S_size): self.policy[s] = np.random.choice(self.A_size, p=prob_policy[s,:])
        else:
            self.prob_policy = 0 * self.prob_policy
            for s, a in self.policy.items(): self.prob_policy[s,a] = 1
        
    
    def extract_policy(self):
        '''
        从效用值中提取出策略。
        '''
        
        V = self.V.copy()
        P = self.P
        rewards = self.rewards
        gamma = self.gamma
        policy = self.policy
        
        ''' Element-wise form. '''
        
        for s in range(self.S_size):
            temp = []
            for a in range(self.A_size):
                q = np.dot(P[a, s, :], rewards[a, s, :] + gamma * V)
                temp.append(q)
            # policy[s] = np.argmax(np.array(temp))                            #FIXME: 默认选择第一个最大动作
            policy[s] = np.random.choice(
                [idx for idx, q in enumerate(temp) if np.abs(q - max(temp)) < 1e-23]
            )
        
        ''' Vector form. '''
        
        # policy_vec = np.diag(np.argmax(np.einsum('ijk,ikl->ijl', P, np.transpose(rewards, [0,2,1]) + gamma*V.reshape((-1,1))), axis=0))     
        # for s in range(self.S_size): policy[s] = policy_vec[s]               #FIXME: 默认选择第一个最大动作
            

    def init_policy_and_V(self, random_init=False, seed=None):
        
        if seed is not None: self.set_seed(seed)
        
        self.V = np.zeros((self.S_size, ), dtype=np.float32)      # Initialize V-values.
        self.Q = np.zeros((self.S_size, self.A_size), dtype=np.float32)
        self.policy = {state: 0 for state in range(self.S_size)}  # Initialize policy.
        self.prob_policy = np.zeros((self.S_size, self.A_size), dtype=np.float32)
        
        if random_init:
            self.V = np.random.uniform(size=(self.S_size, ))
            self.Q = np.random.uniform(size=(self.S_size, self.A_size))
            self.policy = {state: random.randint(0, self.A_size-1) for state in range(self.S_size)}
            self.prob_policy = np.random.uniform(size=(self.S_size, self.A_size))
            self.prob_policy = self.prob_policy / np.sum(self.prob_policy, axis=1, keepdims=True)
            
    
    def init_policy_with_Q_table(self, Q_table):
        
        self.policy = {state: np.argmax(Q_table[state]) for state in range(self.S_size)}
        
    
    def set_seed(self, seed):
        
        random.seed(seed)
        np.random.seed(seed)
    
    
    def set_policy(self, policy):
        
        self.policy = policy
        
    
    def set_V(self, V):
        
        self.V = V
        
    
    def random_choose_state(self, weight=None):
        
        if not weight:
            return random.randint(0, self.S_size-1)
        return random.choices(list(range(self.S_size)), weights=weight, k=1)[0]
    
    
    def random_choose_action(self, weight=None):
        
        if not weight:
            return random.randint(0, self.A_size-1)
        return random.choices(list(range(self.A_size)), weights=weight, k=1)[0]
        
        
    def sample_with_epsilon_greedy_policy(self, state, epsilon=.05, max_length=1000, terminate_state=None,
                                          return_terminate_state=False):
        '''
        用epsilon-greedy策略来从某个状态开始采样。
        参数：
            state: 采样开始的初始状态 -> int.
            epsilon:  -> float.
            max_length: 采样得到的链条的最大长度
            terminate_state: 终止状态集合 -> list.
            return_terminate_state: 是否返回轨迹的最终状态？如果True的话，s_list和a_list的长度会+1. -> bool.
        '''
        
        def sample_one_step(state, action):
            '''
            Given a (s, a), get the next state s' and r(s, a, s').
            '''
            # Get the prob.
            trans_prob = self.P[action, state].tolist()
            # Sample next state.
            next_state = self.random_choose_state(weight=trans_prob)
            # Now we can get the reward of last move.
            last_reward = self.rewards[action, state, next_state]
            
            return (last_reward, next_state)
        
        def sample_one_action(state):
            '''
            Given a state s, use epsilon-greedy policy to get the action and its prob.
            '''
            # Get the policy.
            action_ori = self.policy[state]
            # Epsilon!
            p = random.random()
            if p > (epsilon - epsilon / self.A_size):
                return action_ori, (1 - epsilon + epsilon / self.A_size)
            else:
                while True:
                    random_action = self.random_choose_action()
                    if random_action != action_ori: break
                return random_action, epsilon / self.A_size
            
        s_list = []
        a_list = []
        r_list = []
        prob_list = []
        
        # Start sampling!
        length = 0
        while length < max_length:
            s_list.append(state)
            # Use epsilon greedy to sample an action.
            action, prob = sample_one_action(state)
            # Transform to next state and get the reward.
            reward, next_state = sample_one_step(state, action)
            
            # Record and step forward.
            a_list.append(action)
            prob_list.append(prob)
            r_list.append(reward)
            state = next_state
            length += 1
            
            # If we are now in a terminate state, we break.
            if (terminate_state is not None) and (state in terminate_state):
                if return_terminate_state:
                    s_list.append(state)
                    action, _ = sample_one_action(state)
                    a_list.append(action)
                break
        
        return s_list, a_list, r_list, prob_list, length
    
    
    def MC_on_policy_control(self, max_iter=1000,
                             epsilon=None,
                             max_length=1000,
                             terminate_state=None,
                             need_return=False,
                             seed=None):
        '''
        On-policy MC Learning.
        参数：
            max_iter: 总共进行的循环次数 -> int.
            epsilon: 若为None将用1/sqrt(k) -> float.
            max_length: 每条轨迹的最长长度 -> int.
            terminate_state: 终止状态集合 -> list.
            need_return: 是否需要输出V_list -> bool.     
            seed: 设置随机数种子 -> any 
        '''
        
        if seed is not None: self.set_seed(seed)
        
        N_table, Q_table = np.zeros((self.S_size, self.A_size)),  .1 * np.random.randn(self.S_size, self.A_size)
        self.init_policy_with_Q_table(Q_table)
        
        V_list = []
        for iter in range(max_iter):
            # Randomly choose an initial state.
            init_state = self.random_choose_state()
            # Sample a trajectory.
            new_epsilon = epsilon if epsilon is not None else 1 / ((iter+1) ** (1/2))
            s_list, a_list, r_list, prob_list, length = self.sample_with_epsilon_greedy_policy(init_state,
                                                                                               epsilon=new_epsilon,
                                                                                               max_length=max_length,
                                                                                               terminate_state=terminate_state)
            
            G = 0

            for t in range(length-1, -1, -1):    # From T-1, T-2, ... 0.
                # Check whether (s_t, a_t) appears before.
                # If it didn't appear, we update our policy.
                s_t, a_t = s_list[t], a_list[t]
                if not (s_t, a_t) in set(zip(s_list[:t], a_list[:t])):
                    N_table[s_t, a_t] = N_table[s_t, a_t] + 1
                    Q_table[s_t, a_t] = Q_table[s_t, a_t] + (r_list[t] + self.gamma*G - Q_table[s_t, a_t]) / N_table[s_t, a_t]
                    self.policy[s_t] = np.argmax(Q_table[s_t])
                # Update G.
                G = self.gamma * G + r_list[t]
                    
            # After each iteration, we evaluate our policy.
            if need_return:
                self.evaluate_policy(epsilon=new_epsilon)
                V_list.append(self.V)
        
        if need_return: return V_list  
        
        
    def MC_off_policy_control(self, max_iter=1000,
                              epsilon=None,
                              max_length=1000,
                              terminate_state=None,
                              need_return=False,
                              magic=False,
                              is_break=True,
                              seed=None):
        '''
        Off-policy MC Learning.
        参数：
            max_iter: 总共进行的循环次数 -> int.
            epsilon: 若为None将用1/sqrt(k) -> float.
            max_length: 每条轨迹的最长长度 -> int.
            terminate_state: 终止状态集合 -> list.
            need_return: 是否需要输出V_list -> bool.
            magic: W放在外面还是里面？ -> bool.
            is_break: 用break还是用continue？ -> bool.
            seed: 设置随机数种子 -> any
        '''
        
        if seed is not None: self.set_seed(seed)
        
        N_table, Q_table = np.zeros((self.S_size, self.A_size)),  .1 * np.random.randn(self.S_size, self.A_size)
        self.init_policy_with_Q_table(Q_table)
        
        V_list = []     # To record the qualities.
        for iter in range(max_iter):
            # Randomly choose an initial state.
            init_state = self.random_choose_state()
            # Sample a trajectory.
            new_epsilon = epsilon if epsilon is not None else 1 / ((iter+1) ** (1/2))
            s_list, a_list, r_list, prob_list, length = self.sample_with_epsilon_greedy_policy(init_state,
                                                                                               epsilon=new_epsilon,
                                                                                               max_length=max_length,
                                                                                               terminate_state=terminate_state)
            
            G, W = 0, 1

            for t in range(length-1, -1, -1):    # From T-1, T-2, ... 0.
                # Check whether (s_t, a_t) appears before.
                # If it didn't appear, we update our policy.
                s_t, a_t = s_list[t], a_list[t]
                if not (s_t, a_t) in set(zip(s_list[:t], a_list[:t])):
                    N_table[s_t, a_t] = N_table[s_t, a_t] + 1
                    if magic:
                        Q_table[s_t, a_t] = Q_table[s_t, a_t] + (r_list[t] + self.gamma*G*W - Q_table[s_t, a_t]) / N_table[s_t, a_t]
                    else:
                        Q_table[s_t, a_t] = Q_table[s_t, a_t] + (W*r_list[t] + self.gamma*G*W - Q_table[s_t, a_t]) / N_table[s_t, a_t]
                    self.policy[s_t] = np.argmax(Q_table[s_t])
                # Check the chain. 
                if a_t != self.policy[s_t]:
                    if is_break:
                        break
                    W = 0
                    continue
                # Update G and W.
                G = self.gamma * G + r_list[t]
                W = W * prob_list[t]
                    
            # After each iteration, we evaluate our policy.
            if need_return:
                self.evaluate_policy()
                V_list.append(self.V)
        
        if need_return: return V_list
        
        
    def SARSA(self,
              max_iter=10000,
              terminate_state=None,
              epsilon=None,
              need_return=False,
              step_size=0.1,
              seed=None,
              plot_freq=100):
        '''
        SARSA.
        参数：
            max_iter: 总共进行的循环次数 -> int.
            epsilon: 若为None将用1/sqrt(k) -> float.
            terminate_state: 终止状态集合 -> list.
            step_size: 步长 -> float.
            seed: 设置随机数种子 -> any     
            plot_freq: 多少次iter之后评估一次V值 -> int.   
        '''
        if seed is not None: self.set_seed(seed)
        
        Q_table = .1 * np.random.randn(self.S_size, self.A_size)
        # Q_table = np.zeros((self.S_size, self.A_size))
        self.init_policy_with_Q_table(Q_table)
        gamma = self.gamma
        
        V_list = []     # To record the qualities.
        # Randomly choose one state.
        init_state = self.random_choose_state() 
        while ((terminate_state is not None) and (init_state in terminate_state)):
            init_state = self.random_choose_state()        
        s_t = init_state
        
        for iter in tqdm(range(max_iter), ncols=80):
            
            new_epsilon = epsilon if epsilon is not None else 1 / ((iter+1) ** (1/2))
            s_list, a_list, r_list, prob_list, length = self.sample_with_epsilon_greedy_policy(s_t,
                                                                                               epsilon=new_epsilon,
                                                                                               max_length=2,
                                                                                               terminate_state=terminate_state,
                                                                                               return_terminate_state=True)
            
            s_t, a_t, r_t, s_next_t, a_next_t = s_list[0], a_list[0], r_list[0], s_list[1], a_list[1]
            # update.
            Q_table[s_t, a_t] = Q_table[s_t, a_t] + step_size * (r_t + gamma * Q_table[s_next_t, a_next_t] - Q_table[s_t, a_t])  
            self.policy[s_t] = np.argmax(Q_table[s_t])
            
            # After each update, we evaluate our policy.
            if need_return and iter % plot_freq == 0:
                self.evaluate_policy()
                V_list.append(self.V)  
            
            while ((terminate_state is not None) and (s_next_t in terminate_state)):
                # import pdb; pdb.set_trace()
                s_next_t = self.random_choose_state()
                
            s_t = s_next_t
                
        if need_return: return V_list
        
        
    def Q_learning(self,
                   max_iter=10000,
                   terminate_state=None,
                   epsilon=None,
                   need_return=False,
                   step_size=0.1,
                   seed=None,
                   plot_freq=100):
        '''
        Q-learning.
        参数：
            max_iter: 总共进行的循环次数 -> int.
            epsilon: 若为None将用1/sqrt(k) -> float.
            terminate_state: 终止状态集合 -> list.
            step_size: 步长 -> float.
            seed: 设置随机数种子 -> any     
            plot_freq: 多少次iter之后评估一次V值 -> int.   
        '''
        if seed is not None: self.set_seed(seed)
        
        Q_table = .1 * np.random.randn(self.S_size, self.A_size)
        # Q_table = np.zeros((self.S_size, self.A_size))
        self.init_policy_with_Q_table(Q_table)
        gamma = self.gamma
        
        V_list = []     # To record the qualities.
        # Randomly choose one state.
        init_state = self.random_choose_state() 
        while ((terminate_state is not None) and (init_state in terminate_state)):
            init_state = self.random_choose_state()        
        s_t = init_state
        
        for iter in tqdm(range(max_iter), ncols=80):
            
            new_epsilon = epsilon if epsilon is not None else 1 / ((iter+1) ** (1/2))
            s_list, a_list, r_list, prob_list, length = self.sample_with_epsilon_greedy_policy(s_t,
                                                                                               epsilon=new_epsilon,
                                                                                               max_length=2,
                                                                                               terminate_state=terminate_state,
                                                                                               return_terminate_state=True)
            
            s_t, a_t, r_t, s_next_t, a_next_t = s_list[0], a_list[0], r_list[0], s_list[1], a_list[1]
            # update.
            Q_table[s_t, a_t] = Q_table[s_t, a_t] + step_size * (r_t + gamma * Q_table[s_next_t].max() - Q_table[s_t, a_t])  
            self.policy[s_t] = np.argmax(Q_table[s_t])
            
            # After each update, we evaluate our policy.
            if need_return and iter % plot_freq == 0:
                self.evaluate_policy()
                V_list.append(self.V)  
            
            while ((terminate_state is not None) and (s_next_t in terminate_state)):
                # import pdb; pdb.set_trace()
                s_next_t = self.random_choose_state()
                
            s_t = s_next_t
                
        if need_return: return V_list        
                                  
    
    def compute_delta(self):
        '''
        计算Delta值
        (请在self.V和self.policy都已经收敛的情况下调用)
        Delta = \min_{s, a\not\in A^*_s} |A(s,a)|
              = \min_{s, a\not\in A^*_s}  Q(s,a^*) - Q(s,a)
        '''
        
        Q_table = np.zeros((self.S_size, self.A_size), dtype=np.float64)
        
        # Naive loop.
        for s in range(self.S_size):
            for a in range(self.A_size):
                Q_table[s, a] = np.dot(self.P[a, s, :], self.rewards[a, s, :] + self.gamma * self.V)
        
        abs_A = self.V.reshape((-1,1)) - Q_table
        abs_A[abs_A < 1e-13] = 1e+15
        Delta_s = np.min(abs_A, axis=1)
        Delta = np.min(Delta_s)
        
        return Delta
    
            
if __name__ == '__main__':
    
    # One test example for debugging.
    P = np.array([
        [
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1]
        ],
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0]
        ]
    ])

    rewards = np.array([
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1]
        ],
        [
            [0, 0, 0],
            [20, 0, 0],
            [0, 0, 0]
        ]
    ])
    
    gamma = .9
    
    one_mdp = MDP(P, gamma, rewards)
    
    one_mdp.value_iteration()
    import pdb; pdb.set_trace()
    one_mdp.init_policy_and_V()
    one_mdp.policy_iteration()
    import pdb; pdb.set_trace()
    one_mdp.init_policy_and_V()
    one_mdp.MC_off_policy_control()
    import pdb; pdb.set_trace()