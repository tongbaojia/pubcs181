#Q-Learning, alternative solutions to deal with uncertain gravity during the
#first steps of each epoch.

# Imports.
from __future__ import division
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey

#Note: the game tends to crash. No error is given, the kernel just dies. I was
#not able to figure out why. Might be the script or a python/anaconda/spyder version issue, unsure.


class Learner(object):
    '''
    This agent uses Q-learning
    '''

    def __init__(self):
        #initialize state, action and reward
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.counter = 0
        
        #initialize values from SwingyMonkey (for some reason I was not able to import them)
        self.screen_height = 400
        self.screen_width  = 600
        self.monkey_height = 57
        
        #define learning parameters
        self.gamma = 0.1
        self.epsilon = 0.01
        self.eta = 0.1
        
        #I considered different sets of states. Because the gap between the upper
        #and the lower part of the tree is a constant, and because the top and the
        #bottom of the screen to not move, it is enough to only keep track of the
        #position of the monkey, its speed, the distance to the next tree, and the
        #height of the top of the lower part of the tree.
        #define the bins to use to define the states
        #Position of the monkey
        self.N_bins_PM = 10
        #Speed of the monkey
        self.N_bins_SM = 10
        #Distance to the next tree
        self.N_bins_DT = 10
        #Height of the top of the lower part of the tree
        self.N_bins_TLT = 10
        #Height of the bottom of the upper part of the tree
        #self.N_bins_BUT = 10
        #Upper gap, between the head of the monkey and the bottom of the upper tree
        #self.N_bins_UG = 10
        #Lower gap between the feet of the monkey and the top of the lower tree
        #self.N_bins_LG = 10
        
        
        #define a reasonnable range of speeds for the monkey 
        self.SM_min = -30
        self.SM_max = + 30
        
        #initialize the gravity to value 0, which correspond to unknown gravity.
        #We can train 5 different models: one for unknown gravity (trained on the
        #first step(s)), and 4 for gravities 1-4.
        self.gravity = 0
        self.confirmed_gravity = False
        
        
        #initialize the Q matrix: dimensions = position monkey, speed monkey,
        # distance to the next tree, height of the top of the lower part of the
        # tree, height of the bottom of the upper part of the tree, gravity,
        #action.
        #self.Q = np.zeros(self.N_bins_PM, self.N_bins_SM, self.N_bins_DT,
        #                  self.N_bins_TLT, self.N_bins_BUT, 4, 2)
        
        #initialize the Q matrix: dimensions = position monkey, speed monkey,
        # distance to the next tree, upper gap, lower gap, gravity, action
        self.Q = np.zeros(shape = (self.N_bins_PM, self.N_bins_SM, self.N_bins_DT,
                          self.N_bins_TLT, 5, 2))


    #define function to distribute each continuous value into the right bucket:
        
    def bin_PM(self, PM):
        return int(np.trunc((PM/self.screen_height)*self.N_bins_PM))
    
    def bin_SM(self, SM):
        if SM < self.SM_min:
            return 0
        elif SM > self.SM_max:
            return self.N_bins_SM -1
        else:
            return int(np.trunc(((SM-self.SM_min)/(self.SM_max - self.SM_min))*(self.N_bins_SM -2))) - 1
        
    def bin_DT(self, DT):
        return int(np.trunc(((DT)/self.screen_width)*self.N_bins_DT))
    
    def bin_TLT(self, TLT):
        return int(np.trunc((TLT/self.screen_height)*self.N_bins_TLT))
    
    #def bin_BUT(self, BUT):
    #    return int(np.trunc((BUT/self.screen_height)*self.N_bins_BUT))
    
    #def bin_UG(self, MLP, TLT):
    #    return int(np.trunc(((TLT-MLP)/(self.screen_height-self.monkey_height))*self.N_bins_UG))

    #def bin_LG(self, MUP, BUT):
    #    return int(np.trunc(((MUP-BUT)/(self.screen_height-self.monkey_height))*self.N_bins_LG))
    
    #the idea is to start using the values of Q associated with the most likely
    # gravity, even before the monkey stops jumping, by correcting
    #the gravity value for the impulse (assuming it took its mean value, 15).
    #However as long as the monkey is jumping, because there is incertitude in
    #the measure, one keep trying to learn the gravity.
    #it might make a difference in helping the monkey survive the second time
    #step if it jumped during the first one.
    #note that this is an alternative to the use of a 5th gravity state (unknown)
    #since with this model only the first step has unknown gravity, and no learning is possible.
    def learn_gravity(self, last_state, last_action, state):
        if last_action == 0:
            confirmed_gravity = True
            gravity = last_state["monkey"]["vel"] - state["monkey"]["vel"]
        else:
            confirmed_gravity = False
            gravity = 15 - last_state["monkey"]["vel"]
        return gravity, confirmed_gravity
    
    def access_Q(self, state):
        PM = self.bin_PM(state['monkey']['bot'])
        SM = self.bin_SM(state['monkey']['vel'])
        DT = self.bin_DT(state['tree']['dist'])
        TLT = self.bin_TLT(state['tree']['bot'])
        #UG = self.bin_UG(state['monkey']['vel'], state['tree']['bot'])
        #LG = self.bin_LG(state['monkey']['vel'], state['tree']['top'])
        G = self.gravity
        return self.Q[PM, SM, DT, TLT, G, :]
    
    def update_Q(self, state, action, update):
        PM = self.bin_PM(state['monkey']['bot'])
        SM = self.bin_SM(state['monkey']['vel'])
        DT = self.bin_DT(state['tree']['dist'])
        TLT = self.bin_TLT(state['tree']['bot'])
        #UG = self.bin_UG(state['monkey']['vel'], state['tree']['bot'])
        #LG = self.bin_LG(state['monkey']['vel'], state['tree']['bot'])
        G = self.gravity
        self.Q[PM, SM, DT, TLT, G, action] -= self.eta*update

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.counter = 0
        self.gravity = 0

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.
        
        #start estimating the gravity, but keep reassessing it until a reliable measure
        #can be obtained. (when the monkey is not jumping)
        if not self.confirmed_gravity:
            if self.last_state is not None:
                self.gravity, self.confirmed_gravity = self.learn_gravity(self.last_state, self.last_action, state)

        #exploration/exploitation trade-off: epsilon greedy algorithm
        if npr.rand() < self.epsilon:
            if npr.rand() < 0.1:
                new_action = 0
            else:
                new_action = 1
            
        else:
            new_action = np.argmax(self.access_Q(state))
            
        new_state  = state
        
        if self.last_state != None: 
            self.Q_variation = self.access_Q(self.last_state)[self.last_action]
            - (self.last_reward + self.gamma*np.max(self.access_Q(new_state)))
        
            self.update_Q(self.last_state, self.last_action, self.Q_variation)

        self.last_action = new_action
        self.last_state  = new_state
        self.counter += 1
        
        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
        
    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games. 
	run_games(agent, hist, 50, 10)

	# Save history. 
	np.save('hist',np.array(hist))


