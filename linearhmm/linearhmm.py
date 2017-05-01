
import random
import string
import sys

import numpy as np

from utils import get_discrete_value
import score as sc

class HMMchain():
    """ 
    This class allows for easy generation of a simulation data set through a given Hidden Markov Model. 
    
    Parameters
    --------
    states: array
        all unique possible states, coded in integers 
    observations: array 
        all unique possible observations, coded in integers
    start_probability: array
        initial probability to be in a certain state
    transition_probability: matrix
        matrix[i][j] is the probability to change from state i to state j
    emission_probability: matrix
        matrix[i][j] is the probability to observe j at state i 
    
    """
    
    def __init__(self, states, observations, start_probability, transition_probability, emission_probability):
        self.states = states
        self.observations = observations
        self.start_probability = start_probability
        self.transition_probability = transition_probability
        self.emission_probability = emission_probability
    
    def simulate_data(self, N):
        """
        
        Parameters
        --------
        N: int
            number of observations to genearte
            
        Return
        --------
        a tuple of 2 lists,
            hidden: hidden states 
            visible: observations corresponding to each hidden state 
        
        """
        
        # intialize variables 
        hidden = []
        visible = []
        
        # generate initial state
        start_state = get_discrete_value(self.states, self.start_probability)
        hidden.append(start_state)
        
        # generate N states and observations 
        for i in range(N):
            current_state = hidden[i]
            
            # generate a hidden state, indexed at i+1
            next_state = get_discrete_value(self.states, self.transition_probability, current_state)
            hidden.append(next_state)
            
            # generate an observation, based on the current hidden state, indexed at i
            observation = get_discrete_value(self.observations, self.emission_probability, current_state)
            visible.append(observation)
                    
        # remove the last state, which does not have corresponding observation
        hidden.pop()
        return (hidden, visible)
    

class multinomialHMM():

    """
    Base class for Hidden Markov Models.

    This class allows for fitting a HMM, estimating probability

    Parameters
    --------

    observations: array
        all unique possible observations, coded in integers

    """

    def __init__(self, observations):
        self.observations = observations

    def decode(self, states, start_probability, transition_probability, emission_probability):

        """
        Find the most likely sequence of states given a HMM and a sequence of observations.
        Implemented using Viterbi algorithm.

        Parameters
        --------
        states: array
            all unique possible states, coded in integers

        start_probability: array
            initial probability to be in a certain state

        transition_probability: matrix
            matrix[i][j] is the probability to change from state i to state j

        emission_probability: matrix
            matrix[i][j] is the probability to observe j at state i

        Return
        --------
        path[state]:array
            the most possible sequence of states 
        """
        
        # intialize variables 
        V = [{}]
        path = {}

        # Initialize for the 1st observations 
        for _s in states:
            V[0][_s] = start_probability[_s] * emission_probability[_s][self.observations[0]]
            path[_s] = [_s]

        # Decode for the rest observations
        for _o in range(1, len(self.observations)):
            V.append({})
            
            # initialize a new path 
            newpath = {}

            for _s in states:
                (prob, state) = max((V[_o-1][s0] * transition_probability[s0][_s] * emission_probability[_s][self.observations[_o]], s0) for s0 in states)
                V[_o][_s] = prob
                newpath[_s] = path[state] + [_s]

            # update path 
            path = newpath


        (prob, state) = max((V[_o][_s], _s) for _s in states)
        
        return (prob, path[state])


    

    def learn(self, num_states, num_obs):
        """
        Learn the transimition matrix and emission matrix of a HMM, given a sequence of
        self.observations and the number of states/self.observations.

        This is implemented by the Baum-welch algorithm.

        Parameters
        --------
        num_states: int
            number of states

        num_obs: int
            number of integers

        Return
        --------
        transition_probability: matrix
            matrix[i][j] is the probability to change from state i to state j

        emission_probability: matrix
            matrix[i][j] is the probability to observe j at state i
        """
        # initialize variables 
        _temp = np.ones((num_states, num_states)) 
        transition_probability = _temp / np.sum(_temp,1)[:, None]
        _temp = np.ones((num_states, num_obs)) 
        emission_probability = _temp/ np.sum(_temp,1)[:, None]
        theta = np.zeros((num_states, num_states, self.observations.size))

        # loop until meeting the condition
        while True:
            _old_t = transition_probability
            _old_e = emission_probability
            transition_probability = np.ones((num_states, num_states))
            emission_probability = np.ones((num_states, num_obs))

            # run forward-backward algorithm to get probabilites 
            _prob = np.zeros((num_states,self.observations.size+1))
            _forw = np.zeros((num_states,self.observations.size+1))
            _backw = np.zeros((num_states,self.observations.size+1))
            sc.score(_old_t, _old_e, self.observations, _prob, _forw, _backw)

            # get transitional probabilities at each time step
            for _i in range(num_states):
                for _j in range(num_states):
                    for _o in range(self.observations.size):
                        theta[_i,_j,_o] = _forw[_i,_j] * _backw[_j,_o+1] * _old_t[_i,_j] * \
                                          _old_e[_j, self.observations[_o]]


            # generate transition_probability and emission_probability
            for _i in range(num_states):
                for _j in range(num_states):
                    transition_probability[_i, _j] = np.sum(theta[_i, _j, :] ) / np.sum(_prob[_i,:])

            transition_probability = transition_probability / np.sum(transition_probability,1)[:, None]

            for _i in range(num_states):
                for _o in range(num_obs):
                    _o_index_right = np.array(np.where(self.observations == _o))+1
                    emission_probability[_i, _o] = np.sum(_prob[_i,_o_index_right])/ np.sum(_prob[_i,1:])

            emission_probability = emission_probability / np.sum(emission_probability,1)[:, None]

            # check if meeting the condition
            if np.linalg.norm(_old_t-transition_probability) < .00001 and np.linalg.norm(_old_e-emission_probability) < .00001:
                break

        return transition_probability, emission_probability