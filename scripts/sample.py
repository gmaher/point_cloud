import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# image = "/home/marsdenlab/vascular_data/OSMSC0110/OSMSC0110-cm.mha"
# UPPER = 175
# LOWER = 45

image = "/home/marsdenlab/vascular_data/cabg5/cabg5_scan2.mha"
UPPER = 400
LOWER = 200

MEAN = (UPPER+LOWER)*1.0/2
WIDTH = (UPPER-LOWER)*1.0/2

image = sitk.ReadImage(image)

im = sitk.GetArrayFromImage(image)

probs = np.exp( -( (im-MEAN)**2)*0.5/(WIDTH**2))*1.0/np.sqrt(2*np.pi*WIDTH**2)

p_image = sitk.GetImageFromArray(probs)
sitk.WriteImage(p_image,'p.mha')

p_max = np.amax(probs)
p_min = np.amin(probs)

p_scaled = (probs-p_min)/(p_max-p_min)

p_image = sitk.GetImageFromArray(p_scaled)
sitk.WriteImage(p_image,'p_scaled.mha')

H,W,D = im.shape

rands = np.random.rand(H,W,D)

samp = (p_scaled > rands).astype(float)

p_image = sitk.GetImageFromArray(samp)
sitk.WriteImage(p_image,'samp.mha')
