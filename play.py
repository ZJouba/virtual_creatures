# rules={"X": "X[-FFF][+FFF]FX",
#        "Y": "YFX[+Y][-Y]", }
import numpy as np


rules={"X": ["X", "XF"],
       "Y": ["Y", "YF"], }

lstring = "Y"

lstring = ''.join([np.random.choice(rules.get(c, [c])) for c in lstring])

rules.get("Y", "Y")


rules["Y"]