import gym
import keras
import heapq
import random
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Model
from keras.layers import Input, Dense,Conv2D
from keras.optimizers import Adam

'''
keras implementation of PRIORITIZED EXPERIENCE REPLAY
Using gym environment cartpole

reference code at https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py#L18-L86
'''


class SumTree(object):
    """
    This SumTree code is modified version and the original code is from: 
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story the data with it priority in tree and data frameworks.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \ 
          1     2
         / \   / \ 
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
    
    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32),[], np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i]= idx
            b_memory.append(data)
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)




class  DoubleDQN_agent:

    def __init__(self,env):
        self.env=env
        self.state_size=env.observation_space.shape[0]
        self.action_size=env.env.action_space.n
        self.gamma=0.95
        self.epsilon=1.0
        self.epsilon_min=0.01
        self.epsilon_decay_r = 0.995
        self.learning_rate = 0.01
        self.model=self._build_model()
        self.target_model=self._build_model()
        self.steps=0
        self.update_target_freq =1000

        self.Memory = Memory(1000)
        

    
    def _build_model(self):

        Inputs= Input(shape=(self.state_size,))
        X= Dense(24,activation='relu')(Inputs)
        X= Dense(24,activation='relu')(X)
        X= Dense(self.action_size)(X)
    
        model = Model(inputs=Inputs, outputs=X)
        model.compile(loss='mae',optimizer=Adam(lr=self.learning_rate, clipvalue=1))

        return model

    def update_model(self):

        if self.steps % self.update_target_freq ==0:
            self.target_model.set_weights(self.model.get_weights())
    
    def compute_target_value(self,next_state, reward, done):
        
        if not done:
            return reward + self.gamma * self.target_model.predict(next_state)[0][np.argmax(self.model.predict(next_state)[0])]
        else:
            return reward

    def compute_TDerror(self, state, action, reward, next_state, done):
        
        y=self.model.predict(state)[0][action]
        y_target=self.compute_target_value(next_state,reward,done)
        return np.abs(y-y_target)

    def remember(self ,state, action, reward, next_state, done):
        
        self.Memory.store((state, action, reward, next_state, done))
    

    def replay(self,batch_size):
    

        tree_idx, batch_memory, ISWeights = self.Memory.sample(batch_size)
        States=np.empty((batch_size,self.state_size))
        Labels=np.empty((batch_size,self.action_size))
        abs_errors=np.empty((batch_size,1))
        abs_errors
        i=0
        for state,action,reward,next_state,done in batch_memory:
    
            target=self.compute_target_value(next_state,reward,done)

            label=self.model.predict(state)
            label[0][action]= target
            States[i,:]=state
            Labels[i,:]=label
            abs_errors[i]=np.sum(np.abs(self.model.predict(state)[0][action]-target),axis=0)
            i+=1

        self.Memory.batch_update(tree_idx, abs_errors)
        self.model.fit(States, Labels , epochs=1, verbose=0, sample_weight=np.squeeze(ISWeights))
        self.epsilon_decay()
        self.update_model()

        return


    def act(self,state):
        
        self.steps+=1
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
    
        act_values = self.model.predict(state)
        
        return np.argmax(act_values[0]) 


    def epsilon_decay(self):
        if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay_r
            
        return




if __name__ =="__main__":

    def train(agent, Episodes=1000, batch_size = 32):

        done=False
        agent.env._max_episode_steps = None
        Score=[]
        for e in range(Episodes):
            state = np.expand_dims(agent.env.reset(), axis=0)
            
            for time in range(1000):
                #agent.env.render()

                action = agent.act(state)
                next_state, reward, done, _ = agent.env.step(action)
                reward = reward if not done else -10
                next_state = np.expand_dims(next_state, axis=0)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    print("episode: {}/{}, score: {}, epsilon: {:.2}".format(e, Episodes, time, agent.epsilon))
                    Score.append(time)
                    break
                if agent.steps > 1000:
                    agent.replay(batch_size)
                    
        return Score
    Env=gym.make('CartPole-v0')
    agent=DoubleDQN_agent(Env)

    score=train(agent)
    plt.plot( score)
    plt.ylabel('score')
    plt.xlabel('episodes')
    plt.title('Double DQN with PRM performance ')
    plt.show()