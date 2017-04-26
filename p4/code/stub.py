# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.cur_state   = None
        self.cur_action  = None
        self.last_reward = None
        
        ##learner private parameters; pretend we don't know them
        self.Value   = 0
        self.screen_width  = 600
        self.screen_height = 400
        self.gravity = 0
        self.edge    = 0

        ##set the descretize values; min; max; binsize
        self.par_tree_dis   = [-100, 600, 50] ##no negative for now
        self.par_monkey_vel = [-50,  50,  10]
        self.par_monkey_top = [0,    450, 50] ##monkey x top
        self.par_diff_top   = [-100, 440, 20] ##monkey x - tree top
        self.par_diff_bot   = [-200, 200, 20] ##monkey x - tree bot

        Q_dimension = self.tran_state({"tree":{"dist": 600, "top": 0, "bot": 0}, 
            "monkey":{"vel":50, "top": 400, "bot": 200}}, init=True)##initialize the Q dimension
        print "initial", Q_dimension
        self.Q = np.zeros(Q_dimension + (2, ))
        ##change intitialization; reward intial jump
        self.Q[ 0, :,  :,   :]   =  [0, 0.1]
        self.Q[ :, 4:, :,   :]  =  [0.1, 0]
        self.Q[ :, :,  :4,  :]    =  [0, 0.1]

        ##set the learning parameters
        self.eta = 0.9
        self.gamma = 0.9
        self.epsilon = 0.01

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.cur_state   = None
        self.cur_action  = None
        self.last_reward = None
        
        ##learner private parameters
        self.Value = 0
        self.gravity = 0
        #self.edge    = 10

    def zoom_value(self, value, info, init=False):
        ''' info is the list: min, max, bin_size; convert input value
            into coded descrete bin information.'''
        if value > info[1]: ##avoid overflow
            value = info[1] - info[2]
        elif value < info[0]: ##avoid underflow
            value = info[0]
        convert = int(np.floor((value - info[0])/info[2]))
        if init:
            return convert + 1
        else:
            return convert

    def tran_state(self, state, init=False):
        '''Transforms the state into a vector, could be felxible.'''
        state_vector = ()
        #state_vector += (self.zoom_value(state["tree"]["dist"], self.par_tree_dis, init), )
        state_vector += (self.zoom_value(state["monkey"]["vel"], self.par_monkey_vel, init), )
        state_vector += (self.zoom_value(state["monkey"]["top"], self.par_monkey_top, init), )
        #state_vector += (self.zoom_value(state["monkey"]["top"] - state["tree"]["top"], self.par_diff_top, init), )
        state_vector += (self.zoom_value(state["monkey"]["bot"] - state["tree"]["bot"], self.par_diff_bot, init), )
        return state_vector

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        def tran_model(state, eta=0.9, gamma=0.9):
            '''return the action, given the state'''

            rel_top = state["tree"]["top"] - state["monkey"]["top"]
            rel_bot = state["monkey"]["bot"] - state["tree"]["bot"]
            rel_vel = state["monkey"]["vel"]

            ##figure out jump velocity
            if self.last_action is not None:
                if self.last_action is True: ##if jumped
                    print rel_vel - self.last_state["monkey"]["vel"]
                else: ## figure out gravity of this world
                    self.gravity =  rel_vel - self.last_state["monkey"]["vel"]

            # if state["monkey"]["top"] +  rel_vel  > 400:
            #     #print "will hit edge bottom"
            #     return False
            ##this works for large gravity well, because the poisson mean is small
            ##let's take gravity into account
            #rel_time = int(state["tree"]["dist"] / 25)
            ##tranlate to 3 steps
            if rel_top - rel_vel  < self.edge: ##do not jump in this case for sure
                #print "will hit tree top"
                return 0
            elif rel_bot + rel_vel  < self.edge + 40/self.gravity:
                #print "will hit tree bottom"
                return 1
            elif state["monkey"]["bot"] -  rel_vel  < self.edge:
                #print "will hit edge bottom"
                return 1
            #print rel_top, rel_bot, rel_vel
            return 0

        #new_action = npr.rand() < 0.2
        #print "new_action:", new_action, "new_state:", state, 
        #print " Value: ", self.Value
        
        if npr.random() < self.epsilon:
            #new_action = npr.choice([0, 1])
            new_action = tran_model(state)
        else:
            #print self.tran_state(state), state
            new_action = np.argmax(self.Q[self.tran_state(state)])

        
        new_action = tran_model(state)
        # def update_value(reward):
        #     if reward < 0:
        #         self.edge += npr.choice([-1,1]) * 1
        #         print "update edge: ", self.edge
        #     return

        # update_value(self.last_reward)
        #new_action = tran_model(state)

        ##update the values
        self.last_action = self.cur_action
        self.last_state  = self.cur_state
        self.cur_action  = new_action
        self.cur_state   = state

        return self.cur_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        #print "reward:", reward

        ### update the Q function
        if self.last_action is not None:
            s_l = self.tran_state(self.last_state)
            s_c = self.tran_state(self.cur_state)
            action = (self.last_action, )
            self.Q[s_l + action] += self.eta * (reward + self.gamma * np.argmax(self.Q[s_c]) - self.Q[s_l + action])

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
        
        #print "action_callback: ", learner.action_callback
        #print "reward_callback: ", learner.reward_callback

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)
        print ii, " mean is:", np.mean(np.array(hist)), "max is: ", max(hist)

        # Reset the state of the learner.
        learner.reset()
        
    return


if __name__ == '__main__':

    # Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = []

    # Run games. 
    run_games(agent, hist, 20, 0)
    print hist
    #print np.mean(agent.Q), agent.Q
    # Save history. 
    np.save('hist',np.array(hist))


