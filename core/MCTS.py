import numpy as np
import random
from tqdm import tqdm
from copy import deepcopy

class Node():
    
    def __init__(self,
                 state,
                 parent,
                 pre_action):
        '''
        初始化一个MCTS节点，对应于MDP中的一个state
        参数：
            state: 
                状态，用idx表示。
            parent:
                父母，一个Node实例 (或者None)
            pre_action:
                父母是经过哪个动作转移到这个节点里的？(或者None)
        '''
        self.state = state
        self.parent = parent
        self.is_leaf = True
        self.pre_action = pre_action
        
        self.actions = []       # A list for actions.
        self.children = []      # A list for nodes.
        self.N = []             # A list for visit counts.
        self.Q = []             # A list for action value.
        self.children_n = 0
        
        node = self
        depth = 0
        while node.parent:
            depth += 1 
            node = node.parent
        self.depth = depth        
        
        
    def _load_transition_table(self,
                               node,
                               transition_table):
        node.N, node.Q = transition_table[node.state]
        
    def _update_table(self,
                      node,
                      transition_table):
        transition_table[node.state] = [deepcopy(node.N), deepcopy(node.Q)]
        
    def move(self,
             s, a, P):
        
        return np.argmax(P[a, s, :])  #FIXME: Only support determinstic model.
        
    def expand(self,
               P,
               gamma,
               rewards,
               terminate_state_list,
               transition_table,
               V_list):
        '''
        Expand this node.
        Return the value of this state.
        '''
        # 1. Check terminal situation.
        if not self.is_leaf:
            raise Exception("This node has been expanded.")
        self.is_leaf = False   
        
        # 2. Propagate.
        v = V_list[self.state]
        node = self       
        while node.parent is not None:
            a = node.pre_action
            s_prime = node.state
            node = node.parent
            s = node.state
            node.N[a] += 1
            v = (gamma * v + rewards[a, s, s_prime])
            node.Q[a] = v / node.N[a] +\
                            node.Q[a] * (node.N[a] - 1) / node.N[a]   
            self._update_table(node, transition_table)
            
        if self.state in terminate_state_list:  
            self.is_leaf = True  
            return 
        
        # 3. Expand this node.
        A_size = P.shape[0]
        s = self.state
        self.N = [0 for _ in range((A_size))]
        self.Q = [0 for _ in range((A_size))]        
        
        self._load_transition_table(self, transition_table)  # Get record from table.
        
        for a in range(A_size):
            child_state = self.move(s, a, P)
            # import pdb; pdb.set_trace()
            child_depth = self.depth + 1
            child_node = Node(
                state = child_state,
                parent = self,
                pre_action = a
            )
            self.children.append(child_node)
        self.children_n = len(self.children)
          
        
    def select(self, c=1):
        '''
        Choose the best child.
        Return the chosen node.
        '''
        if self.is_leaf:
            raise Exception("Cannot choose a leaf node.")
        
        scores = [self.Q[i] + c * np.sqrt(np.log((sum(self.N))+1) / (self.N[i]+1))
                  for i in range(self.children_n)]
        return self.children[np.argmax(scores)], scores
        

class MCTS():
    
    def __init__(self,
                 P,
                 gamma,
                 rewards,
                 terminate_state_list,
                 V_list):
        '''
        初始化一个MCTS模型。
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
            termintate_state_list:
                用于记录终止状态的列表。
            V_list:
                用于记录最终策略的价值。
        注意：
            状态空间S和动作空间A按照默认的顺序标号方式表示，已暗含在参数P和rewards中。
        '''
        
        assert P.shape == rewards.shape, "P和rewards的形状应该一致"
        self.P = P
        self.gamma = gamma 
        self.rewards = rewards
        
        self.A_size, self.S_size, _ = self.P.shape
        assert self.A_size > 1 and self.S_size > 1, "动作空间和状态空间不能大小为1"
        
        self.terminate_state_list = terminate_state_list
        self.V_list = V_list
        
        # Transition table.
        A_size = P.shape[0]
        S_size = P.shape[1]
        N = [0 for _ in range((A_size))]
        Q = [0 for _ in range((A_size))]  
        self.transition_table = {
            state: [deepcopy(N), deepcopy(Q)] for state in range(S_size)
        }
        
        
    def run(self,
            root_state,
            max_step,
            c):
        '''
        Run MCTS.
        '''
        V_list = []
        # Select a leaf node.
        root_node = Node(state=root_state,
                    parent=None,
                    pre_action=None)        
        for step in range(max_step):
            node = root_node
            while not node.is_leaf:
                node, scores = node.select(c)
            # print(self.transition_table)
            node.expand(self.P,
                        self.gamma,
                        self.rewards,
                        self.terminate_state_list,
                        self.transition_table,
                        self.V_list)
            
            
            # import pdb; pdb.set_trace()
            one_V_array = [
                np.max(_[1]) for _ in self.transition_table.values()
            ]
            V_list.append(one_V_array)
            
        # Return the policy.
        return self.transition_table, V_list