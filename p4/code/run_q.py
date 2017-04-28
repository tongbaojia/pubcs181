# CS 181 
# Q learning with epsilon-greedy policy.

# Each state is described by:
# - clearance between monkey and top tree
# - clearance between monkey and ground
# - horizontal clearance between monkey and next tree
# - velocity
# - gravity

# All distances are rounded to the nearest coarse grid multiple to shrink state space

# Imports.
import numpy as np
import numpy.random as npr        
import  matplotlib.pyplot as plt

from SwingyMonkey import SwingyMonkey

class Learner(object):

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.last_vel = None
        self.count = 0
        
        self.epsilon = 0.0 # probability that we explore at every step
        self.gamma = 0.99 #discount for future reward
        self.eta = 0.01 #step for gradient descent
        
        #coarse grid parameters to reduce number of states
        self.grid_size = 100
        self.vel_grid_size= 5
        
        #These are dimensions of the state space
        self.n_top_gaps = 800/self.grid_size
        self.n_front_gaps = 600/self.grid_size
        self.n_ground_gaps = 400/self.grid_size
        self.n_speeds = 200/self.vel_grid_size
        self.n_grav = 2 #number of possible values of gravity

        self.Q = np.zeros((self.n_top_gaps, self.n_front_gaps,self.n_ground_gaps, self.n_speeds, self.n_grav, 2))
        
        self.default_grav = 1
        self.grav = None

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.last_vel = None
        
        self.count = 0
        
        self.grav = None

    def action_callback(self, state):
        # Return 0 to swing and 1 to jump.
        # Epsilon-greedy policy
    
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
                    self.grav = (self.last_vel-vel)/4
                    grav = self.grav
        else:
            grav = self.grav
        
        #Write down current state
        current_state=(top_gap/self.grid_size+self.n_top_gaps/2, front_gap/self.grid_size,ground_gap/self.grid_size,  vel/self.vel_grid_size+self.n_speeds/2, grav-1) 
        
        #Update Q(s,a) from previous (s,a) using current s',a'
        if self.last_state is not None:
            self.Q[self.last_state][self.last_action]-= self.eta*(self.Q[self.last_state][self.last_action]-self.last_reward-self.gamma*np.max(self.Q[current_state]))  
        
        #Make next move
        if npr.rand()<self.epsilon or self.count == 0:
            #Explore
            #print('exploring')
            if npr.rand <0.5 or self.count == 0:
                new_action = 0
            else:
                new_action = 1
        else:
            #Maximize reward
            q = self.Q[current_state]
            new_action = np.argmax(q)  
            #print('max reward = '+str(np.max(q)))
        self.last_vel = vel    
        self.last_action = new_action
        self.last_state  = current_state
        self.count +=1
        return self.last_action

    def reward_callback(self, reward):
        #Received reward from previous move.
        self.last_reward = reward
        #print('received reward '+ str(reward))


def run_games(learner, hist, grav, iters = 1000, t_len = 2000):
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
    N_iter = 100
    run_games(agent, hist, grav, N_iter, 10)
    
    # Save history. 
    filename = 'eps_'+str(agent.epsilon)+'_g_'+str(agent.gamma)+'_eta_'+str(agent.eta)+'_grid_'+str(agent.grid_size)+'_vgrid_'+str(agent.vel_grid_size)+'_iter_'+str(N_iter)
    
    thefile = open(str(filename)+'.txt', 'w')
    for (score,g) in zip(hist, grav):
        thefile.write("%d\t%d\n" % (score,g))
    thefile.close()
    
    print('Average score = '+str(np.mean(hist)))
    # Plot the data and the regression line.
    plt.plot(hist, 'o')
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.title('Average score = '+str(np.mean(hist)))
    plt.savefig(filename+'.png')

    plt.show()