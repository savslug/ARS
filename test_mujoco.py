import gym
import mujoco_py
from gym import wrappers
from augmented_random_search import AugmentedRandomSearch

policy_params=dict(
    enable_v1=False,
    enable_v2=False,
    enable_t=True,
    n_directions=4,
    n_top_directions=2,
    exploration_noise=1,#mu
    #rollout_length=500

)
#ars=AugmentedRandomSearch('CartPole-v1',policy_params)
ars=AugmentedRandomSearch('Hopper-v3',policy_params)

for i in range(10000):
    ars.evaluate(do_render=True)
    ars.train_step()

"""
t，平均と標準偏差を求めるのにすごく時間がかかっている
"""
