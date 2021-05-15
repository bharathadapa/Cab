# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space =[(0,0)]+[(i,j) for i in range(m) for j in range(m) if i!=j]
        self.state_space = [[x,y,z] for x in range(m) for y in range(t) for z in range(d)]
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch2(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        state_encod=[1 if i==state[0] else 0 for i in range(m)]+[1 if j==state[1] else 0 for j in range(t)]+[1 if k==state[2] else 0 for k in range(d)]
        return state_encod


    # Use this function if you are using architecture-2 
    # def state_encod_arch1(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""
    #     return state_encod


    ## Getting number of requests
    def sam_req(self,location):
        if location == 0:
            requests = np.random.poisson(2)
        elif location==1:
            requests = np.random.poisson(12)
        elif location==2:
            requests = np.random.poisson(4)
        elif location==3:
            requests = np.random.poisson(7)
        elif location==4:
            requests = np.random.poisson(8)
        return requests
    
    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        req=self.sam_req(state[0])
        #while req==0:
        #    req=self.sam_req(state[0])
        
        if req >15:
            req =15
        if req!=0:
            possible_actions_index = [0]+random.sample(range(1, (m-1)*m +1), req) # (0,0) is not considered as customer request
        else:
            possible_actions_index=[0]
        actions = [self.action_space[i] for i in possible_actions_index]

        
        return possible_actions_index,actions   



    def reward_func(self, state, act_in, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        t1,t2,next_loc=self.time_cal(state, act_in, Time_matrix)
        action=self.action_space[act_in]
        
        if act_in!=0:
            #reward=R*(Time_matrix[action[0]][action[1]][state[1]][state[2]])-C*(Time_matrix[state[0]][action[0]][state[1]][state[2]]+Time_matrix[action[0]][action[1]][state[1]][state[2]])
            reward=R*(t2)-C*(t1+t2)
        else:
            reward=-C
        return reward

    def time_cal(self, state, act_in, Time_matrix):
        action=self.action_space[act_in]
        if act_in!=0:
            t1=int(Time_matrix[state[0]][action[0]][state[1]][state[2]])
            t11=state[1]+t1
            if t11<=23:
                t2=int(Time_matrix[action[0]][action[1]][t11][state[2]])
            else:
                d11=(state[1]+t1)//24
                if d11+state[2]<7:
                    t2=int(Time_matrix[action[0]][action[1]][t11%24][state[2]+d11])
                else:
                    t2=int(Time_matrix[action[0]][action[1]][t11%24][(state[2]+d11)%7])
            next_loc=action[1]
        else:
            t1,t2=1,0
            next_loc=state[0]
        return t1,t2,next_loc

    def next_state_func(self, state, act_in, Time_matrix):
        """Takes state and action as input and returns next state"""
        
        t1,t2,next_loc=self.time_cal(state, act_in, Time_matrix)
        action=self.action_space[act_in]
        
        next_time=state[1]+t1+t2
        
        if next_time<=23:
            next_state=[next_loc,next_time,state[2]]
        else:
            next_day=next_time//24
            if (state[2]+next_day)<7:
                next_state=[next_loc,next_time%24,state[2]+next_day]
            else:
                next_state=[next_loc,next_time%24,(state[2]+next_day)%7]
        return next_state,t1,t2




    def reset(self):
        return self.action_space, self.state_space, self.state_init
