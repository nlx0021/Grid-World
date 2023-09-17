import random
import numpy as np
import matplotlib.pyplot as plt
from pylab import xticks, yticks

from MDP import MDP
from random_board import generate_random_board, generate_one_goal_board

class Grid_world():
    
    def __init__(self,
                 board,
                 gamma=.9,
                 win_reward=1,
                 punish_reward=-1):
        '''
        初始化构建一个Grid World游戏。
        参数:
            board:
                输入一个二维数组。0代表空格，1代表障碍而2代表结束方块。
                example:
                    [[1, 0, 0],
                     [0, 0, 2]]
            gamma:
                MDP折扣因子。
            win_reward:
                走到终点格子的奖励值。
            punish_reward:
                走到障碍格子的惩罚值(为负数)。
        注意：
            五个动作已经按照默认排序。按照“上下左右停”来排序。
        '''
        
        # print("Building Grid world!")
        
        self.board = board.astype(np.uint8)
        self.H, self.W = board.shape
        self.gamma = gamma
        self.win_reward = win_reward
        self.punish_reward = punish_reward
        
        # Use (i, j) to present a state first.
        self.state_list = [(i, j) for i in range(self.H) for j in range(self.W)]
        self.pos2idx = {(i, j): i * self.W + j for i in range(self.H) for j in range(self.W)}
        self.idx2pos = dict([val,key] for key,val in self.pos2idx.items())
        # Use a string to represent an action.
        self.action_list = ["up", "down", "left", "right", "stay"]
        self.action2idx = {"up": 0, "down": 1, "left": 2, "right": 3, "stay": 4}
        self.idx2action = dict([val,key] for key,val in self.action2idx.items())
        
        # Build the MDP.
        self.P, self.rewards = self.load_board()
        self.mdp = MDP(self.P, self.gamma, self.rewards)
        
        
    def load_board(self):
        '''
        从输入的board中建立起状态空间、动作空间、奖励模型以及状态转移模型。
        '''
        
        def move(s, a):
            '''
            从状态s执行动作a，得到的下一个状态是？
            '''
            if a == "up":
                return s if s[0] == 0 else (s[0]-1, s[1])
            elif a == "down":
                return s if s[0] == (self.H - 1) else (s[0]+1, s[1])
            elif a == "left":
                return s if s[1] == 0 else (s[0], s[1]-1)
            elif a == "right":
                return s if s[1] == (self.W - 1) else (s[0], s[1]+1)
            else:
                return s
            
        def find_pos(board, element):
            '''
            找到对应元素的位置。
            '''
            result = []
            i_list, j_list = np.where(board == element)
            for idx in range(i_list.size):
                result.append((i_list[idx], j_list[idx]))
            return result
            
        
        board = self.board
        H, W = self.H, self.W
        pos2idx = self.pos2idx
        action2idx = self.action2idx
        
        # 读取目标方块和障碍方块
        target_state_list = find_pos(board, 2)
        obstacle_state_list = find_pos(board, 1)
        
        self.target_state_list = target_state_list
        self.obstacle_state_list = obstacle_state_list
        
        # 状态转移模型P
        P = np.zeros((5, H*W, H*W))
        for a in self.action_list:
            for s in self.state_list:
                next_s = move(s, a)
                P[action2idx[a], pos2idx[s], pos2idx[next_s]] = 1
        for target_state in target_state_list:
            P[:, pos2idx[target_state], :] = 0
            P[:, pos2idx[target_state], pos2idx[target_state]] = 1               # 无论采取什么动作，最终方格永远循环
        
        # 价值函数rewards
        rewards = np.zeros((5, H*W, H*W))
        for target_state in target_state_list:
            # 对所有能够到达胜利格子的设置胜利奖励
            rewards[:, :, pos2idx[target_state]] = self.win_reward * (P[:, :, pos2idx[target_state]] == 1).astype(np.float32)
            rewards[:, pos2idx[target_state], pos2idx[target_state]] = 0         # 排除自我循环
        
        for obstacle_state in obstacle_state_list:
            # 对所有能够到达障碍格子的设置惩罚
            rewards[:, :, pos2idx[obstacle_state]] = self.punish_reward * (P[:, :, pos2idx[obstacle_state]] == 1).astype(np.float32)
            rewards[:, pos2idx[obstacle_state], pos2idx[obstacle_state]] = 0     # 排除自我循环
            
        return P, rewards
    
    
    def solve_mdp(self,
                  mode="value_iteration",
                  max_iter=10000,
                  epsilon=1e-7,
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
            asynchoronous:
                是否采用异步更新 (仅针对value iteration)
            init:
                是否在求解前将MDP的V和policy进行初始化？
            verbose:
                是否进行plt的show输出？
            need_return:
                是否返回V_list？
        '''
        
        assert mode in ["value_iteration", "policy_iteration"], "Param mode must be 'value_iteration' or 'policy_iteration'!"
        
        if init:
            self.mdp.init_policy_and_V(random_init=True)
        
        if not verbose:
            print("Solving!")
        
        if mode == "value_iteration":
            V_list = self.mdp.value_iteration(epsilon=epsilon, max_iter=max_iter, asynchronous=asynchronous,
                                              need_return=need_return, silence=verbose)
        else:
            V_list = self.mdp.policy_iteration(max_iter=max_iter,
                                              need_return=need_return, silence=verbose)
        
        if not verbose:
            self.visualize_policy()
            
        if need_return:
            return V_list
        
    
    def solve_mdp_using_MC_Learning(self,
                                    mode="Off-policy",
                                    max_iter=1000,
                                    epsilon=None,
                                    max_length=10000,
                                    verbose=False):
        '''
        使用MC policy Control的model free方式进行Grid World的(近似)求解。
        参数：
            mode: Off-policy or On-policy -> str.
            max_iter: 最大的迭代数 -> int.
            epsilon: 若为None将用1/sqrt(k) -> float.
            max_length: 每个轨迹的最大长度 ->  int.
            magic_test: 测试magic -> bool.
            break_test: 测试break -> bool.
            verbose: 是否plot -> bool.
        '''
        assert mode in ["Off-policy", "On-policy"]
        
        # 首先用传统方法求解MDP，得到baseline。
        self.solve_mdp(verbose=verbose)
        baseline_V = self.mdp.V
        
        print("Solving!")
        
        target_state_idx_list = [self.pos2idx[target_state] for target_state in self.target_state_list]
        if mode == "Off-policy":
            V_list = self.mdp.MC_off_policy_control(max_iter=max_iter,
                                                    epsilon=epsilon,
                                                    max_length=max_length,
                                                    terminate_state=target_state_idx_list,
                                                    need_return=True,
                                                    seed=1)
        else:
            V_list = self.mdp.MC_on_policy_control(max_iter=max_iter,
                                                   epsilon=epsilon,
                                                   max_length=max_length,
                                                   terminate_state=target_state_idx_list,
                                                   need_return=True,
                                                   seed=1)            
        
        # 绘制V曲线
        V_array = np.stack(V_list, axis=0)
        V_mean_curse = np.mean(V_array, axis=1).tolist()
        if not verbose:
            plt.plot(V_mean_curse)
            plt.axhline(np.mean(baseline_V), color='red', alpha=.7)            
            plt.show()
        plt.clf()
        
        if not verbose:
            self.visualize_policy() 
        
        return V_mean_curse, np.mean(baseline_V)
    
    
    def solve_mdp_using_TD_Learning(self,
                                    mode="Off-policy",
                                    max_iter=1000,
                                    step_size=.1,
                                    epsilon=None,
                                    verbose=False,
                                    plot_freq=100):
        '''
        使用TD的model free方式进行Grid World的(近似)求解。
        参数：
            mode: SARSA or Q-learning -> str.
            max_iter: 最大的迭代数 -> int.
            epsilon: 若为None将用1/sqrt(k) -> float.
            verbose: 是否plot -> bool.
            step_size: 步长 -> float.
        '''
        assert mode in ["SARSA", "Q-learning"]
        
        # 首先用传统方法求解MDP，得到baseline。
        self.solve_mdp(verbose=verbose)
        baseline_V = self.mdp.V
        
        print("Solving!")
        
        target_state_idx_list = [self.pos2idx[target_state] for target_state in self.target_state_list]
        if mode == "SARSA":
            V_list = self.mdp.SARSA(max_iter=max_iter,
                                    epsilon=epsilon,
                                    step_size=step_size,
                                    terminate_state=target_state_idx_list,
                                    need_return=True,
                                    seed=1,
                                    plot_freq=plot_freq)
        else:
            # raise NotImplementedError
            V_list = self.mdp.Q_learning(max_iter=max_iter,
                                         epsilon=epsilon,
                                         step_size=step_size,
                                         terminate_state=target_state_idx_list,
                                         need_return=True,
                                         seed=1,
                                         plot_freq=plot_freq)         
        
        # 绘制V曲线
        V_array = np.stack(V_list, axis=0)
        V_mean_curse = np.mean(V_array, axis=1).tolist()
        if not verbose:
            plt.plot(V_mean_curse)
            plt.axhline(np.mean(baseline_V), color='red', alpha=.7)            
            plt.show()
        plt.clf()
        
        if not verbose:
            self.visualize_policy() 
        
        return V_mean_curse, np.mean(baseline_V)    
    
                      
    
    def visualize_policy(self):
        '''
        可视化得到的策略。
        '''
        
        def draw_one_arrow(s, a):
            y, x = s
            y, x = y+.5, x+.5
            
            if a == "up":
                start_x, start_y = x, y+.25
                ax.arrow(start_x, start_y, 0, -.5, length_includes_head=True, head_width=0.1, head_length=0.1, fc = 'r', ec = 'b')
            elif a == "down":
                start_x, start_y = x, y-.25
                ax.arrow(start_x, start_y, 0, .5, length_includes_head=True, head_width=0.1, head_length=0.1, fc = 'r', ec = 'b')
            elif a == "left":
                start_x, start_y = x+.25, y
                ax.arrow(start_x, start_y, -.5, 0, length_includes_head=True, head_width=0.1, head_length=0.1, fc = 'r', ec = 'b')
            elif a == "right":
                start_x, start_y = x-.25, y
                ax.arrow(start_x, start_y, .5, 0, length_includes_head=True, head_width=0.1, head_length=0.1, fc = 'r', ec = 'b')
        
        def draw_one_block(s, c):
            y, x = s
            plt.fill_between([x,x+1], y, y+1, facecolor=c)
                
        
        # 设置初始画布 (i.e. board).
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, self.W); ax.set_ylim(0, self.H)
        xticks(np.linspace(0, self.W-1, self.W, endpoint=True))
        yticks(np.linspace(0, self.H-1, self.H, endpoint=True))
        ax.grid()
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        # 给特殊格点染色
        for target_state in self.target_state_list:
            draw_one_block(target_state, "green")
        for obstacle_state in self.obstacle_state_list:
            draw_one_block(obstacle_state, "red")
        
        # 开始绘制箭头
        policy = self.mdp.policy
        idx2pos = self.idx2pos
        idx2action = self.idx2action
        for s_idx, a_idx in policy.items():
            draw_one_arrow(idx2pos[s_idx], idx2action[a_idx])
        
        plt.show()
        
        
    def plot_V_curve_in_VI(self, plot_num=5, differ_epsilon=1e-2):
        '''
        绘制值迭代过程中，对应的贪心策略的效用值曲线。
        '''
        
        self.mdp.init_policy_and_V(random_init=False)
        
        V_list = self.mdp.value_iteration(need_return=True, silence=True)        
        
        V_policy_list = []
        for V in V_list:
            self.mdp.set_V(V)
            self.mdp.extract_policy()
            self.mdp.evaluate_policy()
            V_policy_list.append(self.mdp.V.copy())
        
        # Plot the curve.
        exception_V_s_curve = []
        for s in range(self.mdp.S_size):
            V_s_curve = [V_policy[s] for V_policy in V_policy_list]
            if not all(x<=(y+differ_epsilon) for x, y in zip(V_s_curve[:-1], V_s_curve[1:])):   # 出现不是单调不减的V_s
                print("发现例外情况！异常状态为: %d" % s)
                exception_V_s_curve.append(V_s_curve)
            plt.plot(V_s_curve)
    
        plt.show()
        
        # 如果出现异常状态，再次绘制其曲线：
        plt.clf()
        for exception_V_s in exception_V_s_curve[:plot_num]:
            plt.plot(exception_V_s)
        
        plt.show()
        
        self.solve_mdp()
        
    
    def print_V(self):
        '''
        输出得到的效用值。
        '''
            
        ct = 0
        for V_value in self.mdp.V:
            print("%8f" % V_value, end='\t')
            ct += 1
            if ct % self.W == 0:
                print("\n")
                
    
    def compute_delta(self):
        '''
        计算Delta值
        '''
        
        Delta = self.mdp.compute_delta()
        
        return Delta
    

if __name__ == '__main__':
    
    # board = np.array([
    #     [0, 0, 0, 0],
    #     [0, 0, 1, 0],
    #     [0, 1, 2, 0],
    #     [0, 1, 0, 0]
    # ], dtype=np.uint8)   
    
    # board = np.array([
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    #     [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # ], dtype=np.uint8)
    
    random.seed(21)
    H = 25; W = 5
    # board = generate_random_board(10, 10, .2, .05)
    board = generate_one_goal_board(H, W)
    
    gamma = .9
    win_reward = 1,
    punish_reward = -10
    
    one_grid_world = Grid_world(board,
                                gamma,
                                win_reward,
                                punish_reward)
    
    one_grid_world.solve_mdp(mode="policy_iteration",
                             init=True)
    import pdb; pdb.set_trace()
    # V_list_1, _ = one_grid_world.solve_mdp_using_MC_Learning(mode="Off-policy", max_iter=10000, epsilon=.1, verbose=True)
    # V_list_2, _ = one_grid_world.solve_mdp_using_MC_Learning(mode="Off-policy", max_iter=10000, epsilon=.5, verbose=True)
    # V_list_3, baseline_V = one_grid_world.solve_mdp_using_MC_Learning(mode="Off-policy", max_iter=10000, epsilon=1, verbose=True)
    
    # plt.plot(V_list_1, color='red')
    # plt.plot(V_list_2, color='black')
    # plt.plot(V_list_3, color='green')
    # plt.axhline(baseline_V, color='blue')
    # plt.show()
    
    # V_list_1, _ = one_grid_world.solve_mdp_using_TD_Learning(mode="SARSA", max_iter=1000000, epsilon=.1, verbose=True, plot_freq=10000)
    # V_list_2, _ = one_grid_world.solve_mdp_using_TD_Learning(mode="SARSA", max_iter=1000000, epsilon=.5, verbose=True, plot_freq=10000)
    # V_list_3, _ = one_grid_world.solve_mdp_using_TD_Learning(mode="SARSA", max_iter=1000000, epsilon=1, verbose=True, plot_freq=10000)
    # V_list_4, _ = one_grid_world.solve_mdp_using_TD_Learning(mode="Q-learning", max_iter=1000000, epsilon=.1, verbose=True, plot_freq=10000)
    # V_list_5, _ = one_grid_world.solve_mdp_using_TD_Learning(mode="Q-learning", max_iter=1000000, epsilon=.5, verbose=True, plot_freq=10000)
    # V_list_6, baseline_V = one_grid_world.solve_mdp_using_TD_Learning(mode="Q-learning", max_iter=1000000, epsilon=1, verbose=True, plot_freq=10000)
    # plt.plot(V_list_1, '-.', color='red', label='SARSA(epsilon=0.1)')
    # plt.plot(V_list_2, '--', color='red', label='SARSA(epsilon=0.5)')
    # plt.plot(V_list_3, '-', color='red', label='SARSA(epsilon=1)')
    # plt.plot(V_list_4, '-.', color='black', label='Q-Learning(epsilon=0.1)')
    # plt.plot(V_list_5, '--', color='black', label='Q-Learning(epsilon=0.5)')
    # plt.plot(V_list_6, '-', color='black', label='Q-Learning(epsilon=1)')    
    # plt.axhline(baseline_V, color='blue')
    # plt.legend()
    # plt.show()
    
    print("State Size: %d" % (H*W))
    
    Delta = one_grid_world.compute_delta()
    print("Delta: %f" % Delta)
    
    Delta_theory = gamma ** (H+W-3) * (1-gamma)
    print("Delta theory: %f" % Delta_theory)
    
    ori_upper_bound = ((H*W)*5 - 5) / (1-gamma) * np.log(1/(1-gamma))
    print("Ori upper bound: %f" % ori_upper_bound)
    
    upper_bound = np.log(2/(Delta_theory * (1-gamma))) / (1-gamma)
    print("Upper bound: %f" % upper_bound)