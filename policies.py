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
        self.enable_v1=policy_params['enable_v1']
        self.enable_v2=policy_params['enable_v2']
        self.enable_t=policy_params['enable_t']
        self.weights = np.zeros((self.ac_dim, self.ob_dim), dtype = np.float64)

    def act(self,ob):
        return np.dot(self.weights,ob)


class LinearPolicyDiscrate(LinearPolicy):
    """
    出力が離散値の場合の方策．
    各選択肢の評価関数として見て，最大の評価値となる選択を出力．
    """

    def act(self,ob):
        return np.dot(self.weights,ob).argmax()

