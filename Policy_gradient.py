import gym
import keras
import random
import numpy as np
from collections import deque
from keras.models import Model
from keras.layers import Input, Dense, Conv2D
from keras.optimizers import Adam

class PG_agent:
    
    def __init__(self,env):
        
        self.env=env
        self.state_size=env.env.observation_space.shape[0]
        self.action_size=env.env.action_space.n
        self.learning_rate=0.01
        self.model=self._build_model()
        self.observations=[]
        self.actions=[]
        self.rewards=[]
        self.gammar=0.9
        
    def _build_model(self):
        
        Inputs= Input(shape=(self.state_size,))
        X=Dense(24,activation="relu")(Inputs)
        X=Dense(24,activation="relu")(X)
        X=Dense(self.action_size,activation="softmax")(X)
        
        model=Model(inputs=Inputs,outputs=X)
        model.compile(loss="categorical_crossentropy",optimizer=Adam(lr=self.learning_rate))
        
        return model
    
    def remember(self ,state, action, reward):
        
        act=np.zeros([self.action_size])
        act[action]=1
        self.observations.append(state)
        self.actions.append(np.array(act).astype('float32'))
        self.rewards.append(reward)
    
    def compute_state_values(self,rewards):
        
        state_values= np.zeros_like(rewards)
        running_add=0
        for t in range(rewards.size)[::-1]:
            running_add= self.gammar*running_add + self.rewards[t]
            state_values[t]=running_add
                       
        return state_values
                    
    def act(self,state):
        
        state=state.reshape([1,state.shape[0]])
        action_prob=self.model.predict(state).flatten()
        prob=action_prob/np.sum(action_prob)
        action=np.random.choice(self.action_size,1,p=prob)[0]
        
        return action
    

    def train(self):

        rewards=np.vstack(self.rewards)
        state_values=self.compute_state_values(rewards)

        X= np.squeeze(np.vstack([self.observations]))
        Y=np.squeeze(np.vstack(self.actions*state_values))

        self.model.fit(X,Y)
        self.observations,self.actions,self.rewards=[],[],[]


if __name__ =="__main__":

    env=gym.make('CartPole-v0')
    env._max_episode_steps = None
    agent=PG_agent(env)
    path='D:/Git/Deep_Learning/Weight/PG/weights.h5'
    try:
        agent.model.load_weights(path)
        print("load weights")
    except:
        print("weights not found")
    episode=0

    while episode!=1000:

        state = agent.env.reset()
        
        for time in range(2000):
            #env.render()
            action=agent.act(state)

            next_state,reward,done,_=env.step(action)
            agent.remember(state,action,reward)
            state=next_state
            if done:
                episode +=1
                agent.train()
                print("Episode:%d  Score:%d"%(episode,time))
                break
        if episode%100 ==0:
            agent.model.save_weights(path)
            print("weights saved!")
