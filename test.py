import gym
from augmented_random_search import AugmentedRandomSearch

policy_params=dict(
    enable_v1=False,
    enable_v2=True,
    enable_t=True,
    n_directions=4,
    n_top_directions=2,

)
ars=AugmentedRandomSearch('Pendulum-v0',policy_params)

for i in range(100):
    ars.evaluate(do_render=True)
    ars.train_step()

"""
t，平均と標準偏差を求めるのにすごく時間がかかっている
"""
