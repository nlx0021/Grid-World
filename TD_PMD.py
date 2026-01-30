import os
import numpy as np

from core.MDP import MDP
from models.Grid_world import Grid_world
from models.Random_model import RandomMDP
from utils.random_board import generate_random_board, generate_one_goal_board

def run_exp(args, model: RandomMDP):
    return_dict = model.TD_PMD(
        samples_N=args["samples_N"],
        actor_step_size_func=args["actor_step_size_func"],
        critic_step_size_func=args["critic_step_size_func"],
        eps_func=args["eps_func"],
        batch_size=args["batch_size"],
        mirror_map=args["mirror_map"],
        use_eps=args["use_eps"],
        adaptive_critic_step_size=args["adaptive_critic_step_size"],
        adaptive_actor_step_size=args["adaptive_actor_step_size"],
        seed=args["seed"],
        init_state=args["init_state"],
        init_action=args["init_action"],
        off_policy=args["off_policy"],
        vartheta=args["vartheta"],
        is_expected=args["is_expected"],
        is_approximated=args["is_approximated"]
    )
    return return_dict

if __name__ == "__main__":
    
    # Test TD on Random MDP.
    S_size = 20
    A_size = 10
    gamma = 0.9
    seed = 21
    
    model = RandomMDP(S_size=S_size, A_size=A_size, gamma=gamma, seed=seed)
    
    # # Test TD on Grid world MDP.
    # H = 10
    # W = 10
    # gamma = 0.9
    # seed = 21
    # p_1 = 0
    # p_2 = 0.2
    
    # board = generate_random_board(H, W, p_1, p_2)
    # model = Grid_world(board=board, gamma=gamma, is_termination=False)
    
    # model.mdp.policy_iteration()
    # model.visualize_policy(path=os.path.join("outputs", "Grid_world_policy.png"))
    # model.print_V()
    # import pdb; pdb.set_trace()
    
    # # Arguments for Alg 1.
    # args_1 = {
    #     "samples_N": 10000,
    #     "critic_step_size_func": lambda n: 1 / (n+1)**0.5,
    #     "actor_step_size_func": lambda n: 10 / (n+1)**(5/6),
    #     "eps_func": lambda n: 1 / (n+1)**(1/6),
    #     "batch_size": 1,
    #     "mirror_map": "npg",
    #     "use_eps": True,
    #     "adaptive_critic_step_size": False,
    #     "adaptive_actor_step_size": False,
    #     "seed": 21,
    #     "init_state": 0,
    #     "init_action": 0, 
    #     "off_policy": False,
    #     "vartheta": 1,
    #     "is_expected": False,
    #     "is_approximated": False
    # }
    
    # return_dict = model.TD_PMD(
    #     samples_N=args_1["samples_N"],
    #     actor_step_size_func=args_1["actor_step_size_func"],
    #     critic_step_size_func=args_1["critic_step_size_func"],
    #     eps_func=args_1["eps_func"],
    #     batch_size=args_1["batch_size"],
    #     mirror_map=args_1["mirror_map"],
    #     use_eps=args_1["use_eps"],
    #     adaptive_critic_step_size=args_1["adaptive_critic_step_size"],
    #     adaptive_actor_step_size=args_1["adaptive_actor_step_size"],
    #     seed=args_1["seed"],
    #     init_state=args_1["init_state"],
    #     init_action=args_1["init_action"],
    #     off_policy=args_1["off_policy"],
    #     vartheta=args_1["vartheta"],
    #     is_expected=args_1["is_expected"],
    #     is_approximated=args_1["is_approximated"]
    # )
    
    # # Plot the results.
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    # plt.plot(return_dict['err_list'], label='Single-loop')
    
    # Arguements for Expected-TD-AC-PMD.
    args_2 = {
        "samples_N": 1000000,
        "critic_step_size_func": lambda n: 1,
        "actor_step_size_func": lambda n: 0.001,
        "eps_func": lambda n: 0,
        "batch_size": 10,
        "mirror_map": "npg",
        "use_eps": False,
        "adaptive_critic_step_size": False,
        "adaptive_actor_step_size": False,
        "seed": 21,
        "init_state": 0,
        "init_action": 0,
        "off_policy": True,
        "vartheta": 0.7,
        "is_expected": True,
        "is_approximated": False
    }
    
    return_dict_2 = run_exp(args_2, model)
    # model.visualize_policy(path=os.path.join("outputs", "Grid_world_policy_By_Expectation.png"))
    # model.print_V()
    # import pdb; pdb.set_trace()
    # Plot the results.
    plt.plot(return_dict_2['err_list'], label='Expected-TD-AC-PMD vartheta=0.7')
    
    # Arguements for Expected-TD-AC-PMD with vartheta=0.
    args_3 = args_2
    args_3["vartheta"] = 0.0
    return_dict_3 = run_exp(args_3, model)
    plt.plot(return_dict_3['err_list'], label='Expected-TD-AC-PMD vartheta=0')
    
    # Arguements for Expected-TD-AC-PMD with vartheta=1.
    args_4 = args_2
    args_4["vartheta"] = 1.0
    return_dict_4 = run_exp(args_4, model)
    plt.plot(return_dict_4['err_list'], label='Expected-TD-AC-PMD vartheta=1')
    
    plt.xlabel("Samples num")
    plt.ylabel("V gap")
    plt.legend()
    plt.grid(True)
    plt.grid(alpha=0.3)
    plt.show()
    plt.savefig(os.path.join("outputs", "Single-loop_vs_Expected_TD-AC-PMD.png"))
    