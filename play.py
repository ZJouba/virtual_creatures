import numpy as np

rules = {"A": {"options": ["AB", "BA"], "probabilities": [0.7, 0.3]},
         "B": {"options": ["A", "B"], "probabilities": [0.5, 0.5]}}

lstring = "A"

lstring = ''.join([np.random.choice(rules.get(c, [c]))
                                     for c in lstring])



c = "A"
rules.get(c, ["Other"])["options"]
rules.get(c, ["Other"])["probabilities"]

np.random.choice(rules.get(c, ["Other"])["options"],
                 p=rules.get(c, ["Other"])["probabilities"])

np.random.ch

["+FX", "-FX"]
len(rules.get("F", ["Other"]))