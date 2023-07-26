import numpy as np

array =np.array([1,2,3,4,5,6,7,8,9,10])

median = np.median(array)
MAD=np.median(np.abs(median-array))

print(MAD)