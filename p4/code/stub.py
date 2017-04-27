# Imports.
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_jump_state  = None
        self.cur_state   = None
        self.cur_action  = None
        self.last_reward = None
        
        ##learner private parameters; pretend we don't know them
        self.Value   = 0
        self.screen_width  = 600
        self.screen_height = 400
        self.gravity = None
        self.edge    = 0
        self.edge_high_offset = 30
        self.edge_low_offset  = 30 ##should actually code learning...

        ##set the descretize values; min; max; binsize
        self.par_tree_dis   = [-100, 400, 100] ##no negative for now; use this
        self.par_monkey_vel = [-50,  50,  5] ##monkey velocity; use this
        self.par_monkey_top = [0,    400, 25] ##monkey x top; use this
        #self.par_diff_top   = [-200, 200, 20] ##monkey x - tree top
        self.par_diff_bot   = [-100, 300, 25] ##monkey x - tree bot; use this

        Q_dimension = self.tran_state({"tree":{"dist": 400, "top": 0, "bot": 0}, 
            "monkey":{"vel":50, "top": 400, "bot": 300}}, init=True)##initialize the Q dimension
        print "initial", Q_dimension
        ##change intitialization; reward intial jump; for high gravity
        self.Q_high = np.zeros(Q_dimension + (2, ))
        self.Q_high[ :, :,   :,  :,   ]    =  [0.02, 0]
        self.Q_high[ :, :5, :,  :,   ]    =  [0, 0.02]##velocity
        self.Q_high[ :, :,   :4, :,   ]   =  [0, 0.02]##monkey top
        self.Q_high[ :, :,   :,  :7,  ]   =  [0, 0.02]##monkey x - tree bot
        ##optimal initalization...
        ##change intitialization; reward intial jump; for low gravity
        self.Q_low = np.zeros(Q_dimension + (2, ))
        self.Q_low[ :, :,  :,  :,  ]     =  [0.02, 0]
        self.Q_low[ :, :4, :, :,   ]     =  [0, 0.02] ##velocity
        self.Q_low[ :, :,  :,  :3, ]     =  [0, 0.02] ##monkey x - tree bot
        self.Q_low[ :, :, 7:,  :,  ]     =  [0.02, 0] ##monkey top
        ##default using Q_high
        #self.Q = self.Q_high

        ##set the learning parameters
        self.iter    = 0
        self.eta     = 0.3
        self.gamma   = 0.9
        self.epsilon = 0.01

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_jump_state  = None
        self.cur_state   = None
        self.cur_action  = None
        self.last_reward = None
        
        ##learner private parameters
        self.gravity = None
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
        state_vector += (self.zoom_value(state["tree"]["dist"], self.par_tree_dis, init), )
        state_vector += (self.zoom_value(state["monkey"]["vel"], self.par_monkey_vel, init), )
        state_vector += (self.zoom_value(state["monkey"]["top"], self.par_monkey_top, init), )
        #state_vector += (self.zoom_value(state["monkey"]["top"] - state["tree"]["top"], self.par_diff_top, init), )
        state_vector += (self.zoom_value(state["monkey"]["bot"] - state["tree"]["bot"], self.par_diff_bot, init), )
        return state_vector

    def the_Q(self):
        #return self.Q_high
        if self.gravity is None:
            if self.last_action == 0:
                self.gravity =  self.last_state["monkey"]["vel"] - self.cur_state["monkey"]["vel"]
                return self.Q_low if self.gravity < 2 else self.Q_high 
            else:
                return self.Q_low
        else:
            #print self.gravity
            return self.Q_low if self.gravity < 2 else self.Q_high 

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        ##decide which Q to actually use and update
        # if self.last_state is not None:
        #     self.gravity =  state["monkey"]["vel"] - self.last_state["monkey"]["vel"]
        # if self.gravity < 2:
        #     self.Q = self.Q_low


        def tran_model(state):
            '''return the action, given the state'''

            rel_top = state["tree"]["top"] - state["monkey"]["top"]
            rel_bot = state["monkey"]["bot"] - state["tree"]["bot"]
            rel_vel = state["monkey"]["vel"]

            ##figure out jump velocity
            if self.last_action is not None:
                if self.last_action is True: ##if jumped
                    print rel_vel - self.last_state["monkey"]["vel"]
                else: ## figure out gravity of this world
                    G =  rel_vel - self.last_state["monkey"]["vel"]

            # if state["monkey"]["top"] +  rel_vel  > 400:
            #     #print "will hit edge bottom"
            #     return False
            ##this works for large gravity well, because the poisson mean is small
            ##let's take gravity into account
            rel_time = int(state["tree"]["dist"] / 25)
            ##tranlate to 3 steps
            safe_low_dist  = self.edge if G > 2 else self.edge + self.edge_low_offset
            safe_high_dist = self.edge if G > 2 else self.edge + self.edge_high_offset
            
            if state["monkey"]["top"] + rel_vel > self.screen_height:
                return 0
            elif rel_top + rel_vel < safe_high_dist: ##do not jump in this case for sure
                #print "will hit tree top"
                return 0
            elif rel_bot + rel_vel < safe_low_dist:
                #print "will hit tree bottom"
                return 1
            elif state["monkey"]["bot"] + rel_vel < safe_low_dist:
                #print "will hit edge bottom"
                return 1
            #print rel_top, rel_bot, rel_vel
            return 0

        #new_action = npr.rand() < 0.2
        #print "new_action:", new_action, "new_state:", state, 
        #print " Value: ", self.Value
        ##iteration dependent epsilon
        self.epsilon = min(0, self.epsilon - self.iter * 0.00001)
        if npr.random() < self.epsilon:
            #new_action = npr.choice([0, 1])
            new_action = tran_model(state)
        else:
            #print self.tran_state(state), state
            Q = self.the_Q()
            new_action = np.argmax(Q[self.tran_state(state)])

        
        # new_action = tran_model(state)
        # def update_value(reward):
        #     if reward < 0:
        #         self.edge += npr.choice([-1,1]) * 1
        #         print "update edge: ", self.edge
        #     return

        # update_value(self.last_reward)
        # new_action = tran_model(state)

        ##update the values
        self.last_action = self.cur_action
        self.last_state  = self.cur_state
        self.cur_action  = new_action
        self.cur_state   = state
        if new_action > 0:
            self.last_jump_state = state

        return self.cur_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        #print "reward:", reward
        self.iter += 1
        Q = self.the_Q()
        ### update the Q function
        if self.last_action is not None:
        #if self.last_jump_state is not None:
            s_l = self.tran_state(self.last_state)
            s_c = self.tran_state(self.cur_state)
            action = (self.last_action, )
            ##or iteration dependent eta:
            self.eta   = max(0.5, self.eta   - self.iter * 0.00001)
            self.gamma = max(0.5, self.gamma - self.iter * 0.00001)
            #print Q[s_l + action]
            Q[s_l + action] += self.eta * (reward + self.gamma * np.max(Q[s_c]) - Q[s_l + action])
            ## correct the last jump state; propagates
            # if self.last_jump_state is not None:
            #     s_j = self.tran_state(self.last_jump_state)
            #     Q[s_j + (1, )] += self.eta * (reward + self.gamma * np.max(Q[s_c]) - Q[s_j + (1, )])
           
        if reward < -1:
            print self.tran_state(self.cur_state), (self.tran_state(self.last_jump_state) if self.last_jump_state is not None else "")
        # ###update the fixed parameters
        # if reward < -5:
        #     self.edge_high_offset += 1
        # elif reward < -1:
        #     self.edge_low_offset  -= 1
        #     #self.edge_high_offset += 1

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
        print ii, " mean is:", np.mean(np.array(hist)), "max is: ", max(hist), "total iter is: ", learner.iter

        # Reset the state of the learner.
        learner.reset()
        
    return


if __name__ == '__main__':

    # Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = []

    # Run games. 
    run_games(agent, hist, 100, 0)
    print hist
    #print agent.Q_high[ :, :,  :,  10,  :]
    print "high g Q: ", np.mean(agent.Q_high), np.mean(agent.Q_high[0]), np.mean(agent.Q_high[1]), np.mean(agent.Q_high[2]), np.mean(agent.Q_high[3])
    print "low g Q: ", np.mean(agent.Q_low), np.mean(agent.Q_low[0]), np.mean(agent.Q_low[1]), np.mean(agent.Q_low[2]), np.mean(agent.Q_low[3])

    
    plt.plot(range(len(hist)), hist, 'o' , label='train')
    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.savefig('../Plot/learn.png', bbox_inches='tight')
    plt.clf()
    #print agent.Q
    #print agent.edge_high_offset, agent.edge_low_offset
    # Save history. 
    np.save('hist',np.array(hist))


