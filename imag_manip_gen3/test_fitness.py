import fitness_ext
import imag_manip_pso as i
import numpy as np
import matplotlib.pyplot as pyplot
pb = i.ParticleBuilder(28, .2)
p = pb.build()
img = i.load_image_as_array('0001cropsmall.jpg')
decoded = np.empty((img.shape[0], img.shape[1], img.shape[2]), dtype = np.int)
d = fitness_ext.img_diff(img, decoded)
print d

