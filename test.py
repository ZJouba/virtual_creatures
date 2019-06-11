from CreatureTools_n import Creature
import numpy as np

choices = [
    'F',
    '+',
    '-',
]

proba1 = np.random.uniform(0, 1)
proba2 = 1 - proba1

rule1 = ''.join([np.random.choice(choices) for _ in range(5)]) + 'X'
rule2 = ''.join([np.random.choice(choices) for _ in range(5)]) + 'X'

params = {
    'num_char': 100,
    'variables': 'X',
    'constants': 'F+-',
    'axiom': 'X',
    'rules': {
        'X': {
            'options': [
                rule1,
                rule2,
            ],
            'probabilities': [proba1, proba2]
        }
    },
    'point': np.array([0, 0]),
    'vector': np.array([0, 1]),
    'length': 1.0,
    'angle': np.random.randint(0, 90)  # random
}

c = Creature(params)
print()
