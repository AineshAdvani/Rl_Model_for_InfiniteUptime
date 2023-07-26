# Import routines

import numpy as np
import math
import random
import itertools
from datetime import datetime


# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger
places=list(range(m)) #[0,1,2,3,4] list of places
hours=list(range(t)) # [0,1,2,3....23] list of hours
day=list(range(d)) #[0,1,2,3,4,5,6] list of days

class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        
        #generate action space
        self.action_space=[(location[0],location[1]) for location in itertools.product(places,places) if location[0]!=location[1]]
        #generate state space
        self.state_space =[(states[0],states[1],states[2]) for states in itertools.product(places,hours,day)]
        #seed with current time to generate a choice closest to pure random
        random.seed(datetime.now())
        self.state_init = random.choice(self.state_space) #chose a random state

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        state_encod=[0]*(m+t+d)
        state_encod[state[0]]=1
        state_encod[m+state[1]]=1
        state_encod[m+t+state[2]]=1
        
        
        return np.array(state_encod)


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0] #current location taken from 0th index of state
        if location == 0:
            requests = np.random.poisson(2)
        elif location == 1:
            requests = np.random.poisson(12)
        elif location == 2:
            requests = np.random.poisson(4)
        elif location == 3:
            requests = np.random.poisson(7)
        elif location == 4:
            requests = np.random.poisson(8)
        
        if requests >15:
            requests =15
        
        actions,possible_actions_index=[],[] #initialize actions and possible actions index
        # assign actions and index only if we get a pick up request
        if requests>0:
            random.seed(datetime.now())
            possible_actions_index = random.sample(range(0, (m-1)*m), requests) # (0,0) is not considered as customer request
            
            actions = [self.action_space[i] for i in possible_actions_index]

        
        actions.append((0,0)) #append offline choice to the list of actions
        possible_actions_index.append(20) #append index of offline choice, which is 20 in this set up
        return possible_actions_index,actions   



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        #we assign the action tuple based on the index received
        if action==20:
            action=(0,0) 
        else:
            action=self.action_space[action]
            
        current_position=state[0] #current position is the 0th index in state
        pickup=action[0] #pick up assigned
        drop=action[1] #drop location assigned
        time_elapsed=0  #initiate time to 0
        
        if action!=(0,0): #do the following only if the action is not equal to offline
            #calculate time in hours for distance between current location to pickup location
            time_btwn_current_position_to_pickup=Time_matrix[current_position][pickup][state[1:]]
            pickup_time=state[1]+time_btwn_current_position_to_pickup
            pickup_date=state[2] #assign date
            
            #increment date if trip time passes 24 hours
            pickup_time,pickup_date=self.update_time(pickup_time,pickup_date)
                        
            time_btwn_pickup_to_drop=Time_matrix[pickup][drop][pickup_time][pickup_date]
            
            #calculate reward
            reward=time_btwn_pickup_to_drop*(R-C)-(time_btwn_current_position_to_pickup*C)
            #calculate time elapsed
            time_elapsed=time_btwn_pickup_to_drop+time_btwn_current_position_to_pickup
        else:
            #if the driver choses to remain offline, time is incremented by 1 hour and reward is -C
            reward=-C
            time_elapsed=1
            
        return reward,time_elapsed




    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
          #we assign the action tuple based on the index received
        if action==20:
            action=(0,0)
        else:
            action=self.action_space[action]
            
        if action!=(0,0):  #do the following only if the action is not equal to offline
            current_position=state[0] #current position is the 0th index in state
            pickup=action[0] #pickup assigned
            drop=action[1] #drop location assigned
            
            #calculate time in hours for distance between current location to pickup location
            time_btwn_current_position_to_pickup=Time_matrix[current_position][pickup][state[1:]]
            
            pickup_time=state[1]+time_btwn_current_position_to_pickup
            pickup_date=state[2] #assign date
            
            #increment date if trip time passes 24 hours
            pickup_time,pickup_date=self.update_time(pickup_time,pickup_date)
                       
           
            time_btwn_pickup_to_drop=Time_matrix[pickup][drop][pickup_time][pickup_date]
            new_place=action[1] #assign drop location
            drop_time=pickup_time+time_btwn_pickup_to_drop
            drop_date=pickup_date
        else:
            #in case driver choses to remain offline, increment time by 1 hour and assign next state accordingly
            time_btwn_current_position_to_pickup=1
            
            pickup_time=state[1] #assign 2nd index of state as pickup time
            new_place=state[0] #assign same place as new place, because the driver chose to be offline
            drop_time=pickup_time+time_btwn_current_position_to_pickup 
            drop_date=state[2] #assign 3rd index of state as pickup date
          
        #increment date if trip time passes 24 hours
        drop_time,drop_date=self.update_time(drop_time,drop_date)
                
        next_state=(new_place,int(drop_time),drop_date)
        
        return next_state




    def reset(self):
        return self.action_space, self.state_space, self.state_init
    
    
    def update_time(self,time,date):
        if time>=24:
            time=int(time-24) #recalculate hours in case the value exceeds 24
            date+=1 # in case the hours exceed 24, increment one day
            if date>=7: #in case days exceed 7, recalculate days
                date=date-7
        return int(time),int(date)
