import numpy as np
import pandas as pd
import gym
import policies
import time
from attrdict import AttrDict

class AugmentedRandomSearch():

    def __init__(self,env_name,policy_params):
        self.env=gym.make(env_name)

        policy_params['ob_dim']=self.env.observation_space.shape[0]
        if self.env.action_space.shape==tuple():
            policy_params['ac_dim']=self.env.action_space.n
            self.policy=policies.LinearPolicyDiscrate(policy_params)
        else:
            policy_params['ac_dim']=self.env.action_space.shape[0]
            self.policy=policies.LinearPolicy(policy_params)

        
        self.hyper_parameters=dict(
            enable_v1=True,
            enable_v2=True,
            enable_t=True,
            learning_rate=0.02,#alpha
            exploration_noise=0.01,#mu
            n_directions=8,#N
            n_top_directions=4,#b
            rollout_length=100
        )
        #self.hyper_parameters=AttrDict(hyp_parameters)

    def _rollout(self,shift=0,rollout_length=None,do_render=False):
        """
        ロールアウトを行なう．
        """
        if rollout_length==None:
            rollout_length=self.hyper_parameters['rollout_length']

        total_reward=0
        steps=0

        ob=self.env.reset()
        for i in range(rollout_length):
            if do_render:
                self.env.render()
            action=self.policy.act(ob)
            #print(action)
            ob,reward,done,_=self.env.step(action)
            steps +=1
            total_reward+=(reward-shift)
            if done:
                break

        return total_reward,steps




    def _aggregate_rollouts(self,evaluate=False):
        """
        ロールアウト結果の集積．
        訓練の場合，方策パラメータの更新差分を出力する．
        """


        #学習時．複数の方向をサンプリングし，それぞれの評価値を取得．
        #それに基づいて更新差分を計算．
        n_directions=self.hyper_parameters['n_directions']

        rollout_result=[]
        for i in range(n_directions):
            modifier=self.hyper_parameters['exploration_noise']
            learning_rate=self.hyper_parameters['learning_rate']
            #デルタをサンプリング
            delta=np.random.normal(0,1,size=self.policy.weights.shape)*modifier
            #デルタの正負2通りでポリシーを設定しロールアウト
            policy_weights_before=self.policy.weights
            self.policy.weights+=delta
            reward_p,_=self._rollout()

            self.policy.weights-=delta
            reward_n,_=self._rollout()
            
            self.policy.weights=policy_weights_before
            rollout_result.append([delta,reward_p,reward_n])

        rollout_result=np.array(rollout_result)
        #更新差分の計算
        evaluation_delta=rollout_result[:,1]-rollout_result[:,2]
        deltas=rollout_result[:,0]
        g_hat=(learning_rate/n_directions)*np.sum(evaluation_delta*deltas)

        return g_hat

        
    def train_step(self):
        """
        訓練．方策探索を行なう．
        """
        g_hat=self._aggregate_rollouts()
        self.policy.weights+=g_hat
        return
        
    def evaluate(self,rollout_length=None,do_render=False):
        """
        評価．
        """
        if rollout_length==None:
            rollout_length=self.env.spec.max_episode_steps
        reward, r_steps = self._rollout(shift = 0, rollout_length = rollout_length,do_render=do_render)
        print("reward:",reward,"survived_steps:",r_steps)