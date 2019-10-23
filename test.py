import gym
from augmented_random_search import AugmentedRandomSearch

policy_params=dict(
    enable_v1=True,
    enable_v2=True,
    enable_t=True,
)
ars=AugmentedRandomSearch('CartPole-v0',policy_params)

for i in range(100):
    ars.evaluate(200,do_render=True)
    ars.train_step()
