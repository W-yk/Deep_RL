import gym
import keras
import random
import numpy as np
from collections import deque
from keras.models import Model
from keras.layers import Input, Dense, Conv2D
from keras.optimizers import Adam

class AC_agent:
    
    def __init__(self,env):
        
        self.env=env
        self.state_size=env.env.observation_space.shape[0]
        self.action_size=env.env.action_space.n
        self.learning_rate=0.01
        self.Amodel=self._build_actor_model()
        self.Cmodel=self._build_critic_model()
        self.observations=[]
        self.observations_n=[]
        self.actions=[]
        self.rewards=[]

        self.gammar=0.9
        
    def _build_actor_model(self):
        
        Inputs= Input(shape=(self.state_size,))
        X=Dense(24,activation="relu")(Inputs)
        X=Dense(24,activation="relu")(X)
        X=Dense(self.action_size,activation="softmax")(X)
        
        Amodel=Model(inputs=Inputs,outputs=X)
        Amodel.compile(loss="categorical_crossentropy",optimizer=Adam(lr=self.learning_rate))
        
        return Amodel

    def _build_critic_model(self):

        Inputs=Input(shape=(self.state_size,))
        X=Dense(24,activation="relu")(Inputs)
        X=Dense(24,activation="relu")(X)
        X=Dense(1)(X)

        Cmodel=Model(inputs=Inputs,outputs=X)
        Cmodel.compile(loss="mse",optimizer=Adam(lr=self.learning_rate*2))

        return Cmodel

                    
    def act(self,state):
        
        state=state[np.newaxis,:]
        action_prob=self.Amodel.predict(state).flatten()
        prob=action_prob/np.sum(action_prob)
        action=np.random.choice(self.action_size,1,p=prob)[0]
        
        return action
    
    
    def remember(self ,state,next_state, action, reward):
        
        act=np.zeros([self.action_size])
        act[action]=1
        self.observations.append(state)
        self.observations_n.append(next_state)
        self.actions.append(np.array(act).astype('float32'))
        self.rewards.append(reward)
        
        
    def critic_learn(self):

        next_state=np.squeeze(np.vstack([self.observations_n]))
        next_value=self.Cmodel.predict(next_state)
        Y=np.array(self.rewards).reshape(next_value.shape) + self.gammar*next_value
        X= np.squeeze(np.vstack([self.observations]))


        TD_error= self.Cmodel.predict(X)-Y
        self.Cmodel.fit(X,Y,verbose=0)
        
        return TD_error
    
    def actor_learn(self,TD_error):

        
        X= np.squeeze(np.vstack([self.observations]))
        Y=np.squeeze(np.vstack(self.actions*TD_error))
        self.Amodel.fit(X,Y,verbose=0)
        self.observations,self.observations_n,self.actions,self.rewards=[],[],[],[]
        

if __name__ =="__main__":

    env=gym.make('CartPole-v0')
    env._max_episode_steps = None
    agent=AC_agent(env)
    Apath='D:/Git/Deep_Learning/Weight/AC/Aweights.h5'
    Cpath='D:/Git/Deep_Learning/Weight/AC/Cweights.h5'
    try:
        agent.Amodel.load_weights(Apath)
        agent.Cmodel.load_weights(Cpath)
        print("load weights")
    except:
        print("weights not found")
    episode=0

    while episode!=10000:

        state = agent.env.reset()
        
        for time in range(2000):
            #env.render()
            action=agent.act(state)

            next_state,reward,done,_=env.step(action)
            agent.remember(state,next_state,action,reward)
            state=next_state
            if done:
                episode +=1
                TD_error=agent.critic_learn()
                agent.actor_learn(TD_error)
                print("Episode:%d  Score:%d"%(episode,time))
                break
        if episode%1000 ==0:
            agent.Amodel.save_weights(Apath)
            agent.Cmodel.save_weights(Cpath)
            print("weights saved!")
    