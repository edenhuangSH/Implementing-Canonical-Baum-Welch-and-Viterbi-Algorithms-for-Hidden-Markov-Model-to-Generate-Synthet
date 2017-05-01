
import numpy as np
from utils import get_discrete_value

        
def test_get_discrete_value():
    
    random.seed(1234)
    values = (0, 1, 2, 3)
    probabilities= np.array([0.25, 0.25, 0.25, 0.25])
    
    assert get_discrete_value(values, probabilities) == 3