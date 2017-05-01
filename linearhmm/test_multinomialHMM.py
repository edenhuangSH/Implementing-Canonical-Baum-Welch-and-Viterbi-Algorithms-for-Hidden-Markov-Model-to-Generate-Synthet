
import numpy as np
import pytest
from multinomialHMM import decode

def setUp(self):
    self.observations = [1, 0, 1, 3, 1, 1, 0, 3, 0, 2]
    states, start_probability, transition_probability, emission_probability

    self.states = ('AG_rich', 'CT_rich')
    self.start_probability = {'AG_rich': 0.6, 'CT_rich': 0.4}
    self.transition_probability = {
                                   'AG_rich' : {'AG_rich': 0.8, 'CT_rich': 0.2},
                                   'CT_rich' : {'AG_rich': 0.4, 'CT_rich': 0.6}
                                   }
    self.emission_probability = {
                                   'AG_rich' : {2: 0.5, 1: 0.1, 3: 0.3, 0:0.1},
                                   'CT_rich' : {2: 0.1, 1: 0.4, 3: 0.1, 0:0.4} }
def test_viterbi(self):
    res0, res1 = (2.038431744000001e-08,
                     ['AG_rich',
                      'CT_rich',
                      'CT_rich',
                      'AG_rich',
                      'AG_rich',
                      'CT_rich',
                      'CT_rich',
                      'AG_rich',
                      'AG_rich',
                      'AG_rich'])
    self.assertEqual((res0, res1))