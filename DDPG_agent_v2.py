import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from matplotlib import pyplot as plt


class DDPG_agent:

    def __init__(self, env):
        self.buffer_size = 10000
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1

        self.env = env
        self.sess = tf.Session()
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]

        self.tau = 0.01  # target update factor
        self.s = tf.placeholder(tf.float32, [None, self.state_size], "state")
        self.a = tf.placeholder(tf.float32, [None, self.action_size], "action")
        self.v_ = tf.placeholder(tf.float32, None, "label")
        self.action_grads = tf.placeholder(tf.float32, None, "Critic_grads")
        self.training = tf.placeholder(tf.bool, None, "training_flag")
        
        with tf.variable_scope("eval",reuse=tf.AUTO_REUSE):
            self.v, self.Ctrain, self.grads = self.create_Critic_model("eval")
            self.actions, self.Atrain = self.create_Actor_model("eval")
        with tf.variable_scope("target",reuse=tf.AUTO_REUSE):
            self.v_t = self.create_Critic_model("target")
            self.a_t = self.create_Actor_model("target")

        self.update_op = self.target_update_op()
        self.sess.run(tf.global_variables_initializer())

        self.Memory = deque(maxlen=self.buffer_size)
        tf.summary.FileWriter("logs/", self.sess.graph)

    def create_Actor_model(self, scope, lr=1e-4):
        with tf.variable_scope("Actor"):
            l1 = tf.layers.batch_normalization(self.s, training=self.training, name="BN")
            l2 = tf.layers.dense(l1, 24, tf.nn.relu, name="dense1")
            l3 = tf.layers.dense(l2, 24, tf.nn.relu, name="dense2")
            actions = tf.layers.dense(l3, self.action_size, tf.nn.tanh, name="actions")

        if (scope == "eval"):
            with tf.variable_scope("Actor_train"):
                weights = tf.global_variables(scope + "/Actor")
                grads = tf.gradients(actions, weights, self.action_grads)

                train_op = tf.train.AdamOptimizer(-lr).apply_gradients(zip(grads, weights))
            return actions, train_op
        else:
            return actions

    def create_Critic_model(self, scope, lr=1e-3):
        with tf.variable_scope("Critic"):
            # Different from usual DQN critic network takes in actions as well

            l1 = tf.layers.batch_normalization(self.s, training=self.training, name="State_BN")
            l1_ = tf.layers.batch_normalization(self.a, training=self.training, name="Action_BN")
            l2 = tf.layers.dense(l1, 24, activation=tf.nn.relu, name="State_Dense1")
            l2_ = tf.layers.dense(l1_, 24, activation=tf.nn.relu, name="Action_Dense1")
            l3 = tf.concat([l2, l2_], axis=-1)
            l4 = tf.layers.dense(l3, 24, tf.nn.relu, name="Dense2")
            v = tf.layers.dense(l4, self.action_size, name="Dense3")

        if (scope == "eval"):
            with tf.variable_scope("Critic_train"):
                loss = tf.reduce_mean(tf.square(v - self.v_))
                train_op = tf.train.AdamOptimizer(lr).minimize(loss)
                action_grads = tf.gradients(v, self.a)

            return v, train_op, action_grads
        else:
            return v

    def Critic_learn(self, states, actions, values):

        action_grads, _ = self.sess.run([self.grads, self.Ctrain],
                                        feed_dict={self.s: states, self.a: actions, self.v_: values,
                                                   self.training: True})
        return action_grads

    def Actor_learn(self, states, grads):
        self.sess.run(self.update_op)

        self.sess.run([self.actions, self.Atrain],
                      feed_dict={self.s: states, self.action_grads: grads, self.training: True})

    def target_update_op(self):
        At_parameters = tf.global_variables("target/Actor")
        Ae_parameters = tf.global_variables("eval/Actor")
        Ct_parameters = tf.global_variables("target/Critic")
        Ce_parameters = tf.global_variables("eval/Critic")
        return [[tf.assign(At_p, ((1 - self.tau) * At_p + self.tau * Ae_p)),tf.assign(Ct_p, ((1 - self.tau) * Ct_p + self.tau * Ce_p))] for At_p, Ae_p, Ct_p, Ce_p in
                zip(At_parameters, Ae_parameters,Ct_parameters, Ce_parameters)]

    def predict(self, states, actions):
        return self.sess.run(self.v_t, feed_dict={self.s: states, self.a: actions, self.training: False})

    def OU_process(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)

    def act(self, state):
        action = self.sess.run(self.a_t, feed_dict={self.s: state, self.training: False})
        action += self.epsilon * self.OU_process(action, 0, 1, 0.1)
        self.epsilon -= 0.00001

        return action

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
                label = reward + self.gamma * self.predict(next_state, self.act(next_state))
            else:
                label = reward

            Actions[i, :] = action
            States[i, :] = state
            Labels[i, :] = label
            i += 1

        grads = self.Critic_learn(States, Actions, Labels)
        self.Actor_learn(States, grads)


if __name__ == "__main__":

    def train(agent, Episodes=500, batch_size=32):

        Score = []

        for e in range(Episodes):
            state = np.reshape(agent.env.reset(), [1, agent.state_size])

            R = 0
            for time in range(1000):

                #agent.env.render()
                action = agent.act(state)
                next_state, reward, done, _ = agent.env.step(np.clip(action[0], -1, 1))
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


    Env = gym.make('LunarLanderContinuous-v2')

    Episodes = 500

    agent = DDPG_agent(Env)

    score = train(agent, Episodes)
    plt.plot(range(Episodes), score)
    plt.ylabel('score')
    plt.xlabel('episodes')
    plt.title('DDPG performance ')
    plt.show()