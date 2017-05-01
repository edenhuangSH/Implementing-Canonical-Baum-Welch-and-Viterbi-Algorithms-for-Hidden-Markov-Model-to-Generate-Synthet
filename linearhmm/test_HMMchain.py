
from HMMchain import HMMchain

def setUp(self):
    self.states = ('AG_rich', 'CT_rich')
    self.observations = (0, 1, 2, 3)
    self.start_probability = {'AG_rich': 0.6, 'CT_rich': 0.4}
    self.transition_probability = {
                                   'AG_rich' : {'AG_rich': 0.8, 'CT_rich': 0.2},
                                   'CT_rich' : {'AG_rich': 0.4, 'CT_rich': 0.6}
                                  }
    self.emission_probability = {
                                   'AG_rich' : {2: 0.5, 1: 0.1, 3: 0.3, 0:0.1},
                                   'CT_rich' : {2: 0.1, 1: 0.4, 3: 0.1, 0:0.4}
                                }
    self.N = 10


def test_simulate(self):
    res0, res1 = (['CT_rich',
  'CT_rich',
  'CT_rich',
  'AG_rich',
  'AG_rich',
  'CT_rich',
  'CT_rich',
  'AG_rich',
  'AG_rich',
  'AG_rich'],
 [3, 1, 0, 2, 2, 1, 0, 2, 0, 3])
    self.assertEqual((res0, res1))