import numpy as np

class Policy():

    def __init__(self, policy_params):

        self.ob_dim = policy_params['ob_dim']
        self.ac_dim = policy_params['ac_dim']
        self.weights = np.empty(0)


    def update_weights(self, new_weights):
        self.weights[:] = new_weights[:]
        return

    def get_weights(self):
        return self.weights

    def get_observation_filter(self):
        return self.observation_filter

    def act(self, ob):
        raise NotImplementedError


class LinearPolicy(Policy):

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        self.enable_v2=policy_params['enable_v2']
        self.weights = np.zeros((self.ac_dim, self.ob_dim), dtype = np.float64)

        self.observed_history=[]
        self.ob_mean=0
        self.ob_std=1

    def act(self,ob):
        ob=ob.flatten()
        if self.enable_v2:
            ob=self.normalize_observation(ob)    
        return np.dot(self.weights,ob)

    def normalize_observation(self,ob):
        """
        過去の観測を元に入力を正規化．
        V2効果．
        """
        self.observed_history.append(ob)
        ret=(ob-self.ob_mean)/self.ob_std
        return ret

    def ob_stat_update(self,):
        """
        正規化のためのMeanとStdを更新．
        """
        hist=np.array(self.observed_history)
        self.ob_mean=np.mean(hist,axis=0)
        self.ob_std=np.std(hist,axis=0)+1e-10

        return

class LinearPolicyDiscrate(LinearPolicy):
    """
    出力が離散値の場合の方策．
    各選択肢の評価関数として見て，最大の評価値となる選択を出力．
    """

    def act(self,ob):
        ob=ob.flatten()
        if self.enable_v2:
            ob=self.normalize_observation(ob) 
        return np.dot(self.weights,ob).argmax()
