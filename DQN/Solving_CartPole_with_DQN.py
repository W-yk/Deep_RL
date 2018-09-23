import random
import gym
import numpy as np
from collections import deque
import keras
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam



def build_model(learning_rate,state_size,action_size):

    '''
    Used for Value Function Approximation.
    The input state for CartPole game is quite simple, 
    thus the model I'm using just have two hidden layers with no 
    convolutional layer. 
    '''

    #Build model with keras
    Inputs= Input(shape=(state_size,))
    X= Dense(24,activation='relu')(Inputs)
    X= Dense(24,activation='relu')(X)
    X= Dense(action_size)(X)
    #nonlinearity is not needed here 
    
    model = Model(inputs=Inputs, outputs=X)
    model.compile(loss='mse',optimizer=Adam(lr=learning_rate)) 
    #'mse' stands for mean square error loss, Adam is a common choice of optimizer

    return model


def remember( Memory,state, action, reward, next_state, done):

    '''
    Remerber the current environment infomation by adding it into the Memory que.
    '''
    Memory.append((state, action, reward, next_state, done))
    return Memory

def replay(action_size,state_size,batch_size,Memory,model,gamma):
    
    '''
    Replay means sample data from the Memory and use it to train our model
    after adding labels computed by Bellman equation.
    '''

    minibatch = random.sample(Memory, batch_size)
    States=np.empty((batch_size,state_size))
    Labels=np.empty((batch_size,action_size))
    i=0
    for state,action,reward,next_state,done in minibatch:
    
        if not done:
            # When the game is not finished
            target=reward + gamma * np.amax(model.predict(next_state)[0])
            # Bellman equation to computer the opimal Q-function
        else:
            # When game ended the Q-funtion is just the current reward
            target=reward
        
        label=model.predict(state)
        label[0][action]= target
        # Only introduce loss for the action taken this timestep
        States[i,:]=state
        Labels[i,:]=label
        i+=1
        # construct the minibatch
    
    model.fit(States, Labels , epochs=1, verbose=0)
    # Train the model with each state in the minibatch
    
    return model

def act(state,epsilon,model,env):
    
    '''
    Randomly select an action with a probability of epsilon
    or act to maximize the Q-Function value estimated by the model 
    '''
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    
    act_values = model.predict(state)
    
            
    return np.argmax(act_values[0]) 

def epsilon_update(epsilon, epsilon_min= 0.01, epsilon_decay= 0.995):

    '''
    The agent should be more dependent on our model after train for some time.
    We can do this by decreasing the probability to choose random actions
    '''
    if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            
    return epsilon


def main(weights_path, Episodes=100, batch_size=32, learning_rate = 0.001):
    
    '''
    The main function to put things together
    
    parameters:

    weights_path: the path to load and save the weights
    Episodes: the times for the agent to play the game
    batch_size: minibatch size for the model to train
    learning_rate: the model learning rate
    '''

    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    gamma = 0.95
    epsilon=0.1
    done = False
    Memory = deque(maxlen=2000)
    #initialize the CartPole environment, hyper-parameters and placeholders.

    model= build_model(learning_rate,state_size,action_size) 
    #Build the model
    
    try:
        model.load_weights(weights_path)
        print("load weights succeed")
    except OSError:
        print("no weights found")
    # load previous trained weights
    
    for e in range(Episodes):
        # e stands for the e-th restarted game

        state = np.reshape(env.reset(), [1, state_size])
        # reshape the state for the keras model to accept
        
        for time in range(500):
            # agent takes an action in each time untill done
                
            env.render()
            # update the frame for the game window 
            action = act(state,epsilon,model,env)
            # get the action by calling the funtion above
            epsilon= epsilon_update(epsilon)
            # renew epsilon
            next_state, reward, done, _ = env.step(action)
            # update environment after taking an action, _ stands for the non-needed info 
            reward = reward if not done else -10
            # the agent is design to last long enough in the game, so the reward is negative when done  
            next_state = np.reshape(next_state, [1, state_size])
            Memory= remember(Memory,state, action, reward, next_state, done)
            state = next_state
            # remeber the new environment anddd update state 
            
            if done:
                print("episode: {}/{}, score: {}, epsilon: {:.2}".format(e, Episodes, time, epsilon))
                # defione the score as the number of actions taken
                break
            # restart the game if done 
                
            if len(Memory) > batch_size:
                model=replay(action_size,state_size,batch_size,Memory,model,gamma)
            # train the model after the Memory is enough
                
            
    model.save_weights(weights_path)
    print('weights saved after {} episode'.format(e) )
    #save the model   
    return

main(weights_path='C:/Users/Wyk/Documents/GitHub/Deep_learning/DQN/weights/weights.h5')