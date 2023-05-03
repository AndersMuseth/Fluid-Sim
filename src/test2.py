#%%
filename = "flip_test6.mp4"


# %%
import numpy as np


with open('flip_test6.npy', 'rb') as f:
    positions = np.load(f)
# %%
positions.shape
# %%
