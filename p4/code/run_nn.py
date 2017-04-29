# CS 181
# Learning Swingy Monkey using a neural net basis function
# Record history of past states, actions, rewards, and Q's.
# Limit history size
# Retrain the net after every epoch
# Take weighted average of predicted Q(s,a) and observed [r(s,a) + max gamma*Q(s',a')] at every step

# Imports.
import numpy as np
import numpy.random as npr        
import  matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler  
import warnings

from SwingyMonkey import SwingyMonkey

class Learner(object):

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.last_vel = None
        
        self.epsilon = 0.01 # probability that we explore at every step
        self.gamma = 0.9 #discount for future reward
        self.eta = 0.1
        
        self.default_grav = 1
        self.grav = None

        self.layers = (5,5)
        self.max_iter = 200
        self.max_history = 600
        
        self.s_a_history = np.empty((0,5))
        self.reward_history = np.empty((0))
        self.Q_history = np.empty((0))
        self.count = 0
        
        self.mlp = None
        self.scaler = StandardScaler(with_mean = False)
        
    def reset(self):
        # If we exceeded max history size
        if self.s_a_history.shape[0]>self.max_history:
            self.s_a_history = self.s_a_history[-self.max_history:, :]
            self.reward_history = self.reward_history[-self.max_history:]
            self.Q_history = self.Q_history[-self.max_history:]
            self.epsilon = 0
            
        #Learn Q function using neural network
        if self.mlp is None:
            self.mlp = MLPRegressor(hidden_layer_sizes=self.layers, max_iter=self.max_iter, verbose = False)
        self.scaler.fit(self.s_a_history)
        s_a_scaled = self.scaler.transform(self.s_a_history)
        self.mlp.fit(s_a_scaled, self.reward_history)
        
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.last_vel = None
        self.count = 0
        
        self.grav = None

    def action_callback(self, state):
        # Return 0 to swing and 1 to jump.
        # Cause all warnings to raise exceptions:
        #Find out what state we are in:
        top_gap = (state['tree']['top']-state['monkey']['top'])
        front_gap = state['tree']['dist']
        ground_gap = state['monkey']['bot']
        vel = state['monkey']['vel']

        #Measure gravity
        grav = self.default_grav
        if self.grav is None:
            if self.last_state is not None:
                if self.last_action == 0:
                    self.grav = (self.last_vel-vel)
                    grav = self.grav
        else:
            grav = self.grav
        
        #Write down current state
        current_state=(top_gap, front_gap,  vel, grav) 
        
        #Update Q(s,a) from previous (s,a) using current s',a'
        if self.last_state is not None:
            self.s_a_history = np.vstack((self.s_a_history, np.append(self.last_state, self.last_action)))
            self.reward_history = np.append(self.reward_history, self.last_reward)
            
        if self.count>0:
            if self.mlp is not None:
                state_0 = self.scaler.transform(np.hstack((self.last_state, 0)).reshape(1,-1))
                state_1 = self.scaler.transform(np.hstack((self.last_state, 1)).reshape(1,-1))
                
                #Weight old Q value with (1-eta) weight, and new update with eta weight
                last_s_a = self.scaler.transform(np.hstack((self.last_state, self.last_action)).reshape(1,-1))
                self.Q_history = np.append(self.Q_history, 0)                                
                self.Q_history[-1] += (1-self.eta)*self.mlp.predict(last_s_a)
                self.Q_history[-1] += self.eta*(self.last_reward+self.gamma*np.max((self.mlp.predict(state_0.reshape(1,-1)),self.mlp.predict(state_1.reshape(1,-1)))))
            else:
                self.Q_history = np.append(self.Q_history, self.last_reward)
                
        if self.count >1:
                self.Q_history[-2] += self.gamma*self.last_reward     
            #self.Q_history[-2] += self.gamma*self.last_reward 

        #Make next move
        if npr.rand()<self.epsilon or self.mlp is None or self.count == 0:
            #Explore
            if npr.rand <0.5 or self.count == 0:
                new_action = 0
            else:
                new_action = 1
        else:
            #Maximize reward
            state_0 = self.scaler.transform(np.hstack((current_state, 0)).reshape(1,-1))
            state_1 = self.scaler.transform(np.hstack((current_state, 1)).reshape(1,-1))
            q =(self.mlp.predict(state_0.reshape(1,-1)),
                     self.mlp.predict(state_1.reshape(1,-1)))
            new_action = np.argmax(q)  
            #print('Chosen action = '+str(new_action))
 

        self.last_vel = vel    
        self.last_action = new_action
        self.last_state  = current_state
        self.count+=1
        return self.last_action

    def reward_callback(self, reward):
        #Received reward from previous move.
        self.last_reward = reward
        #print('received reward '+ str(reward))


def run_games(learner, hist, grav,iters = 1000, t_len = 2000):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    print(iters)
    for ii in range(iters):
        #print(ii)
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            state = swing.get_state()
        # Save score history.
        hist.append(swing.score)
        grav.append(learner.grav)

        # Reset the state of the learner.
        learner.reset()      
    return

if __name__ == '__main__':

    # Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = []
    grav = []
    
    # Run games. 
    N_iter= 100
    run_games(agent, hist, grav, N_iter, 1)
    
    # Save history. 
    filename = 'NN_layers'+str(agent.layers)+'eps_'+str(agent.epsilon)+'_g_'+str(agent.gamma)+'_iter_'+str(N_iter)
    
    thefile = open(str(filename)+'.txt', 'w')
    for (score,g) in zip(hist, grav):
        thefile.write("%d\t%d\n" % (score,g))
    thefile.close()
    
    # Plot the data and the regression line.
    plt.clf()
    plt.plot(hist, 'o')
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.title('Average score = '+str(np.mean(hist)))
    plt.savefig(filename+'.png')

    plt.show()
    #np.save('hist',np.array(hist))