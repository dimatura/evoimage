
import numpy as np
import genetic_exp2 as g
import matplotlib.pyplot as pyplot

g.NUM_BLOBS = 3

ind = g.Individual()

# hand draw rectangles
ind.data[0] = np.array([0.0, 0.0, 0.5, 0.5, 0.8, 0.0, 0.0])
ind.data[1] = np.array([0.5, 0.0, 1.0, 0.5, 0.0, 0.8, 0.0])
ind.data[2] = np.array([0.25, 0.25, 0.75, 0.75, 0.0, 0.0, 0.8])

img = g.load_image_as_array('0001cropsmall.jpg')

fiteval = g.FitnessEvaluator(img)

fiteval.cy_decode_data(ind)

dec = fiteval.decoded.copy().astype('uint8')

pyplot.axis('off')
pyplot.imshow(dec)
pyplot.show()
