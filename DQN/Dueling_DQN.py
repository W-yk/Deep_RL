import gym
import keras
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.backend as K
from collections import deque
from keras.models import Model
from keras.layers import Input, Dense,Conv2D,BatchNormalization
from keras.layers.core import Lambda
from keras.optimizers import Adam



class  DoubleDQN_agent:

    def __init__(self,env):
        self.env=env
        self.state_size=env.observation_space.shape[0]
        self.action_size=env.env.action_space.n
        self.Memory=deque(maxlen=2000)
        self.gamma=0.95
        self.epsilon=1.0
        self.epsilon_min=0.01
        self.epsilon_decay_r = 0.995
        self.learning_rate = 0.01
        self.model=self._build_model()
        self.target_model=self._build_model()
        self.steps=0
        self.update_target_freq =1000
    
    def _build_model(self):

        Inputs= Input(shape=(self.state_size,))
        X= Dense(24,activation='relu')(Inputs)
        X= Dense(24,activation='relu')(X)
        X = Dense(self.action_size+1,activation="linear")(X)
        def dueling_layer(X):
            X_V = X[:,0:1]
            X_A = X[:,1:]
            X = X_V + X_A - K.mean(X_A, axis=1, keepdims=True)
            return X
        X = Lambda(dueling_layer, output_shape = (self.action_size,))(X)

        model = Model(inputs=Inputs, outputs=X)
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate, clipvalue= 0.5))

        return model

    def update_model(self):

        if self.steps % self.update_target_freq ==0:
            self.target_model.set_weights(self.model.get_weights())


    def remember(self ,state, action, reward, next_state, done):
        self.Memory.append((state, action, reward, next_state, done))
    
        return

    def replay(self,batch_size):
    
        minibatch = random.sample(self.Memory, batch_size)
        States=np.empty((batch_size,self.state_size))
        Labels=np.empty((batch_size,self.action_size))
        i=0
        for state,action,reward,next_state,done in minibatch:
    
            if not done:
                target=reward + self.gamma * self.target_model.predict(next_state)[0][np.argmax(self.model.predict(next_state)[0])]
            else:
                target=reward
        
            label=self.model.predict(state)
            label[0][action]= target
            States[i,:]=state
            Labels[i,:]=label
            i+=1
        self.model.fit(States, Labels , epochs=1, verbose=0)
        self.epsilon_decay()
        self.update_model()

        return


    def act(self,state):
    
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
    
        act_values = self.model.predict(state)
        self.steps+=1
        
        return np.argmax(act_values[0]) 


    def epsilon_decay(self):
        if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay_r
            
        return


if __name__ =="__main__":

# test on cartpole on openai gym

    def train(agent, Episodes=500, batch_size = 32):

        done=False
        agent.env._max_episode_steps = None
        Score=[]

        for e in range(Episodes):
            state = np.expand_dims(agent.env.reset(), axis=0)

            for time in range(1000):
                agent.env.render()
                action = agent.act(state)
                next_state, reward, done, _ = agent.env.step(action)
                reward = reward if not done else -10
                next_state = np.expand_dims(next_state,axis=0)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    print("episode: {}/{}, score: {}, epsilon: {:.2}".format(e, Episodes, time, agent.epsilon))
                    Score.append(time)
                    break

                if len(agent.Memory) > batch_size:
                    agent.replay(batch_size)

        return Score


    Env=gym.make('CartPole-v0')

    Episodes=500

    agent=DoubleDQN_agent(Env)

    score=train(agent, Episodes)
    plt.plot(range(Episodes), score)
    plt.ylabel('score')
    plt.xlabel('episodes')
    plt.title('Double DQN performance ')
    plt.show()