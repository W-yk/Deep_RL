import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from matplotlib import pyplot as plt
class Critic:

    def __init__(self, sess, n_observation, n_action):
        self.sess = sess
        self.n_O = n_observation
        self.n_A = n_action
        self.tau = 0.01  # target update factor
        self.s = tf.placeholder(tf.float32, [None,self.n_O], "state")
        self.a = tf.placeholder(tf.float32, [None,self.n_A], "action")
        self.v_ = tf.placeholder(tf.float32, None, "BE_value")
        self.training = tf.placeholder(tf.bool, None, "training_flag")
        
        self.v, self.train, self.action_grads = self.create_model("eval")
        self.v_t, _, __ = self.create_model("target")

        self.update_op = self.target_update_op()

        self.sess.run(tf.global_variables_initializer())

    def create_model(self, scope, lr=0.01):
        with tf.variable_scope(scope):
            with tf.variable_scope("Critic"):
                # Different from usual DQN critic network takes in actions as well

                l1 = tf.layers.batch_normalization(self.s,training=self.training, name="State_BN")
                l1_ = tf.layers.batch_normalization(self.a,training=self.training, name="Action_BN")
                l2 = tf.layers.dense(self.s, 24, activation=tf.nn.relu, name="State_Dense1")
                l2_ = tf.layers.dense(self.a, 24, activation=tf.nn.relu, name="Action_Dense1")
                l3 = tf.concat([l2, l2_], axis=-1)
                l4 = tf.layers.dense(l3, 24, tf.nn.relu, name="Dense2")
                v = tf.layers.dense(l4, self.n_A, name="Dense3")

            with tf.variable_scope("C_train"):
                loss = tf.reduce_mean(tf.square(v - self.v_))
                train = tf.train.AdamOptimizer(lr).minimize(loss)
                action_grads = tf.gradients(v, self.a)

        return v, train, action_grads

    def target_update_op(self):
        t_parameters = tf.global_variables("target/Critic")
        e_parameters = tf.global_variables("eval/Critic")

        return [[tf.assign(t_p, ((1 - self.tau) * t_p + self.tau *e_p))] for t_p, e_p in zip(t_parameters, e_parameters)]

    def predict(self, states, actions):

        return self.sess.run(self.v_t, feed_dict={self.s: states, self.a: actions, self.training: False})

    def learn(self, states, actions, values):

        self.sess.run(self.update_op)

        action_grads, _ = self.sess.run([self.action_grads, self.train],
                                             feed_dict={self.s: states, self.a: actions, self.v_: values, self.training: True})
        return  action_grads

class Actor:

    def __init__(self, sess, n_observation, n_action):
        self.sess = sess
        
        self.epsilon = 1
        self.n_O = n_observation
        self.n_A = n_action
        self.tau = 0.01  # target update factor

        self.s = tf.placeholder(tf.float32, [None,self.n_O], "state")
        self.action_grads = tf.placeholder(tf.float32, None, "Critic_grads")
        self.training = tf.placeholder(tf.bool, None, "training_flag")
        self.actions, self.train = self.create_model("eval")
        self.actions_t, __ = self.create_model("target")
        
        self.update_op = self.target_update_op()

        self.sess.run(tf.global_variables_initializer())

    def create_model(self, scope, lr=0.01):
        with tf.variable_scope(scope):
            with tf.variable_scope("Actor"):

                l1 = tf.layers.batch_normalization(self.s,training=self.training, name="BN")
                l2 = tf.layers.dense(self.s, 24, tf.nn.relu, name="dense1")
                l3 = tf.layers.dense(l2, 24, tf.nn.relu, name="dense2")
                actions = tf.layers.dense(l3, self.n_A, tf.nn.tanh,name="actions")

            with tf.variable_scope("A_train"):
                weights = tf.global_variables(scope+"/Actor")
                grads = tf.gradients(actions, weights, self.action_grads)

                train_op = tf.train.AdamOptimizer(-lr).apply_gradients(zip(grads, weights))

        return actions, train_op

    def target_update_op(self):
        t_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target/Actor")
        e_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="eval/Actor")



        return [[tf.assign(t_p, ((1 - self.tau) * t_p + self.tau *e_p))] for t_p, e_p in zip(t_parameters, e_parameters)]

    def OU_process(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)
    
    def predict(self, state):
    
        action= self.sess.run(self.actions_t, feed_dict={self.s : state, self.training:False})
        action+= self.epsilon*self.OU_process(action,0,1,0.1)
        self.epsilon-=0.00001
        
        return action

    def learn(self, states, grads):
        self.sess.run(self.update_op)

        self.sess.run([self.actions, self.train], feed_dict={self.s: states, self.action_grads: grads, self.training: True})



class DDPG_agent:

    def __init__(self, env):

        self.buffer_size = 2000
        self.batch_size = 32
        self.gamma = 0.99

        self.env = env
        self.sess = tf.Session()
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]

        self.actor = Actor(self.sess, self.state_size, self.action_size)
        self.critic = Critic(self.sess, self.state_size, self.action_size)

        self.Memory = deque(maxlen=self.buffer_size)
        tf.summary.FileWriter("logs/", self.sess.graph)
        
    def remember(self, state, action, reward, next_state, done):
        self.Memory.append((state, action, reward, next_state, done))
    
    


    def replay(self, batch_size):

        minibatch = random.sample(self.Memory, batch_size)
    
        States = np.empty((batch_size, self.state_size))
        Actions = np.empty((batch_size, self.action_size))
        Labels = np.empty((batch_size, self.action_size))
        
        i = 0
        
        for state, action, reward, next_state, done in minibatch:

            if not done:
                label = reward + self.gamma * self.critic.predict(next_state, self.actor.predict(next_state))
            else:
                label = reward

            Actions[i, :] = action
            States[i, :] = state
            Labels[i, :] = label
            i += 1

        grads = self.critic.learn(States, Actions, Labels)
        self.actor.learn(States, grads)





if __name__ =="__main__":

# test on cartpole on openai gym

    def train(agent, Episodes=500, batch_size = 32):


        Score=[]

        for e in range(Episodes):
            state = np.reshape(agent.env.reset(), [1, agent.state_size])

            R=0
            for time in range(1000):

                agent.env.render()
                action = agent.actor.predict(state)
                next_state, reward, done, _ = agent.env.step(np.clip(action[0],-1,1))
                reward = reward if not done else -10
                next_state = np.reshape(next_state, [1, agent.state_size])
                
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                R += reward
                if done:

                    print("episode: {}/{}, score: {}".format(e, Episodes, R))
                    Score.append(R)
                    break

                if len(agent.Memory) > batch_size:
                    agent.replay(batch_size)

        return Score


    Env=gym.make('LunarLanderContinuous-v2')

    Episodes=500

    agent=DDPG_agent(Env)

    score=train(agent, Episodes)
    plt.plot(range(Episodes), score)
    plt.ylabel('score')
    plt.xlabel('episodes')
    plt.title('DDPG performance ')
    plt.show()


