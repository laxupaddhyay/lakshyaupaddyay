from joblib import dump,load
import numpy as np
model= load('Dragon.joblib')
input=np.array([[-0.43942006 , 3.12628155 ,-1.12165014, -0.27288841, -1.42262747, -0.24141041,
 -1.31238772,  2.61111401 ,-1.002216859 , -0.5778192  ,-0.97491834 , 0.41164221,
 -0.86091034]])
xx=model.predict(input)
print(xx)