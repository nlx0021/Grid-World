import random
import numpy as np
import matplotlib.pyplot as plt
from pylab import xticks, yticks
from tqdm import tqdm

from core.MDP import MDP
from utils.random_board import generate_random_board, generate_one_goal_board
from utils.utils import *
from models.Grid_world import Grid_world
from models.Random_model import RandomMDP

from run import run

# exp_name = "Exp"
# exp_name = "r-SPG"
exp_name = "Poly"

S_size = 50
A_size = 10
gamma = .9
seed = 21

seed = np.random.randint(65536)
seed = 42
model = RandomMDP(S_size=S_size,
                    A_size=A_size,
                    gamma=gamma,
                    seed=seed)   

if exp_name == "Exp":
    run(
        model=model,
        max_iter=5000,
        param_list=[
            [{"alg": "softmax_NPG", "label": "SNPG"}, 1],
            [{"alg": "phi", "label": "Exp(3,5)", "phi": phi_exp_inv_factory(5, 3)}, 1],
            [{"alg": "phi", "label": "Exp(5,7)", "phi": phi_exp_inv_factory(7, 5)}, 1],
        ],
        metric="rho",
        exp_mode="multi-run",
        noise=None,
        seed=seed
    )

elif exp_name == "Poly":
    step_size = lambda p: (1-model.mdp.gamma) ** 3  / (10 * p ** 2 * model.mdp.A_size ** (2/p))
    run(
        model=model,
        max_iter=20000,
        param_list=[
            [{"alg": "escort_normalized", "label": "EPG(2)", "p": 2}, 0.01],  
            [{"alg": "phi", "label": "Poly(2)", "phi": phi_poly_factory(2), "step_include_d": True}, 0.01],
            [{"alg": "escort_normalized", "label": "EPG(4)", "p": 4}, 0.01],  
            [{"alg": "phi", "label": "Poly(4)", "phi": phi_poly_factory(4), "step_include_d": True}, 0.01],
        ],
        metric="rho",
        exp_mode="multi-run",
        noise=None,
        seed=seed
    )

elif exp_name == "r-SPG":
    run(
        model=model,
        max_iter=10000,
        param_list=[
            [{"alg": "softmax_NPG", "label": "SNPG"}, 1],
            [{"alg": "softmax_adaptive", "label": "reshaped SPG"}, 1],
            [{"alg": "softmax", "label": "SPG"}, 1]
        ],
        metric="rho",
        exp_mode="multi-run",
        noise=None,
        seed=seed
    )