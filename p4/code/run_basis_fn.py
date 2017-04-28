# CS181
# Q-leaning Swingy Monkey using basis functions
# Linear basis was tried, as well as 
# a time basis (where all distances are converted to time to collision)

# Imports.
import numpy as np
import numpy.random as npr        
import  matplotlib.pyplot as plt

from SwingyMonkey import SwingyMonkey

class Learner(object):

    def __init__(self):
        self.last_Q  = None
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.last_vel = None
        self.count = 0
        
        self.epsilon = 0.1# probability that we explore at every step
        self.gamma = 0.9 #discount for future reward
        self.eta = 0.1 #step for gradient descent
        
        # Coefficients for linear basis function
        self.a0_coeff = npr.rand(6)
        self.a1_coeff = npr.rand(6)
        
        # Coefficients for time-based basis functions
        # (Converting distances to times-to-collision using velocity and gravity values)
        """
        self.A = 0
        self.B = 0
        self.C = 0
        self.D = 0
        self.E = 0
        self.F = 0
        self.offset0 = 0
        self.offset1 = 0
        """
        
        #gravity
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
                    self.grav = (self.last_vel-vel)
                    grav = self.grav
        else:
            grav = self.grav
        current_state = np.array((1, top_gap, front_gap, ground_gap, vel, grav))
        
        #Update Q(s,a) from previous (s,a) using current s',a'
        if self.last_state is not None:
            q = (np.dot(self.a0_coeff, current_state), np.dot(self.a1_coeff, current_state))
            if self.last_action ==0:
                self.a0_coeff-=self.eta*(self.last_Q-self.last_reward-self.gamma*np.max(q))*self.a0_coeff
            else:
                self.a1_coeff-=self.eta*(self.last_Q-self.last_reward-self.gamma*np.max(q))*self.a1_coeff
            """
            q = (self.A*vel/grav+
                       self.B*np.sqrt(np.abs(float(vel**2+2*grav*ground_gap)))/grav+self.C*front_gap+self.offset0,
                      self.D*vel/grav+
                       self.E*np.sqrt(np.abs(float(vel**2-2*grav*top_gap)))/grav+self.F*front_gap+self.offset1)     
            if self.last_action ==0:
                self.A-=self.eta*vel/grav*(self.last_Q-self.last_reward-self.gamma*np.max(q))
                self.B-=self.eta*np.sqrt(np.abs(float(vel**2+2*grav*ground_gap)))/grav*(self.last_Q-self.last_reward-self.gamma*np.max(q))
                self.C-=self.eta*front_gap*(self.last_Q-self.last_reward-self.gamma*np.max(q))
                self.offset0-=self.eta*(self.last_Q-self.last_reward-self.gamma*np.max(q))
            else:
                self.D-=self.eta*vel/grav*(self.last_Q-self.last_reward-self.gamma*np.max(q))
                self.E-=self.eta*np.sqrt(np.abs(float(vel**2-2*grav*top_gap)))/grav*(self.last_Q-self.last_reward-self.gamma*np.max(q))
                self.F-=self.eta*front_gap*(self.last_Q-self.last_reward-self.gamma*np.max(q))
                self.offset1-=self.eta*(self.last_Q-self.last_reward-self.gamma*np.max(q))
                """
        #Make next move
        #Write down current state-action q values
        """q = (self.A*vel/grav+
                       self.B*np.sqrt(np.abs(float(vel**2+2*grav*ground_gap)))/grav+self.C*front_gap,
                      self.D*vel/grav+
                       self.E*np.sqrt(np.abs(float(vel**2-2*grav*top_gap)))/grav+self.F*front_gap)"""
        q = (np.dot(self.a0_coeff, current_state), np.dot(self.a1_coeff, current_state))
        
        if npr.rand()<self.epsilon or self.count == 0:
            #Explore
            if npr.rand <0.5 or self.count == 0:
                new_action = 0
            else:
                new_action = 1
        else:
            #Maximize reward
            new_action = np.argmax(q)  
        self.last_vel = vel    
        self.last_action = new_action
        self.last_state  = current_state
        self.last_Q = np.max(q)
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
    N_iter = 50
    run_games(agent, hist, grav, N_iter, 2)
    
    # Save history. 
    filename = 'Basis_eps_'+str(agent.epsilon)+'_g_'+str(agent.gamma)+'_eta_'+str(agent.eta)+'_iter_'+str(N_iter)
    
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
    #np.save('hist',np.array(hist))