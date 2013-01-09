from PIL import Image
import matplotlib.pyplot as pyplot
import numpy as np
import random as rnd

img = Image.open('0001resz.jpg')
imgarr = np.asarray(img).copy()

imgarr = np.bitwise_xor(imgarr, np.roll(imgarr, 50))
imgarr = np.bitwise_or(imgarr, np.roll(imgarr, 30))
imgarr = np.bitwise_xor(imgarr, np.roll(imgarr, 70))
imgarr = np.bitwise_or(imgarr, np.roll(imgarr, 20))
imgarr = np.bitwise_xor(imgarr, np.roll(imgarr, 20))
imgarr = np.bitwise_or(imgarr, np.roll(imgarr, 20))
imgarr = np.bitwise_xor(imgarr, np.roll(imgarr, 80))
#imgarr = np.log(imgarr)
#imgarr = np.exp(imgarr)
#imgarr = np.bitwise_xor(imgarr, np.roll(imgarr, -430))
#imgarr = np.bitwise_or(imgarr, np.flipud(imgarr))
#imgarr = np.bitwise_and(imgarr, np.fliplr(imgarr))
#    #imgarr = np.bitwise_xor(imgarr, np.roll(imgarr, i*10))
    #imgarr = np.bitwise_xor(imgarr, np.fliplr(imgarr))

# MAX_ROW_HEIGHT = 2
# MAX_MAX_REPETITIONS = 200
#
# x_splits = []
# new_split = 0
# #TODO vectorize with cmumsum
# while new_split < imgarr.shape[0]-1:
#     x_splits.append(new_split)
#     new_split += 1 + np.random.binomial(MAX_ROW_HEIGHT, .5)
#
# rows = [r for r in np.array_split(imgarr, x_splits, axis = 0) if r.size > 0]
#
# new_rows = []
# height = 0
# i = 0
# max_max_reps = np.random.binomial(MAX_MAX_REPETITIONS, .5) + 1
# h = imgarr.shape[0] - 1
# while i < len(rows) and height < h:
#     if np.random.sample() < .01:
#         max_max_reps = np.random.binomial(MAX_MAX_REPETITIONS, .8) + 1
#     reps = np.random.poisson(0.01*max_max_reps)
#     #row = np.tile(rows[i], (reps[i], 1, 1))
#     for rep in xrange(reps):
#         #if height < h:
#         height += rows[i].shape[0]
#         new_rows.append(rows[i])
#         #else:
#         #    #new_rows[np.random.random_integers(0, len(new_rows) - 1)] = rows[i]
#         #    new_rows[i%len(new_rows)] = rows[i]
#     i += 1
# print height
# imgarr = np.concatenate(new_rows, axis = 0)
#

pyplot.imshow(imgarr)
pyplot.show()
