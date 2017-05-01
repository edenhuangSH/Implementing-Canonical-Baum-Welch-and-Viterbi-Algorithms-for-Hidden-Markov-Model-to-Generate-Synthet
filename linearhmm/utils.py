import random

def get_discrete_value(values, probabilities, current_state=None):
        """ Generate a random value given a discrete distribution.
        
        Parameters
        ----------
        values: array
            all unique possible values of this distribution
        probabilities: matrix or array
            matrix: matrix[i][j] is the probability to observe j at state i 
            array: array[j] is the probability to observe j
        current_state: int
            current hidden state, encoded in integer; default as None
            
        Returns
        ----------
        value: int
            a realized value of a random variable determined by current_state and its distribution
        
        
        """
        
        # when current_state is None, treate probabilites directly as array
        if current_state is not None:
            probability_array = probabilities[current_state]
        else:
            probability_array = probabilities
        
        r = random.random()
        prev = 0
        for value in values:
            prev += probability_array[value]
            if r < prev:
                return value

def encode_DNA(x):
    
    """convert base to integer representation"""
    
    if x == 'T': 
        num = 0
    elif x == 'C': 
        num = 1
    elif x == 'A': 
        num = 2
    elif x == 'G': 
        num = 3
    else:
        num = 4
    
    return num