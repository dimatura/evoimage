from PIL import Image
import matplotlib.pyplot as pyplot
import numpy as np
import random as rnd

img = Image.open('0001resz.jpg')
imgarr = np.asarray(img).copy()

num_giblets = 2000
max_giblet_size = 3
#max_giblet_size = (30,30)

x0 = np.random.random_integers(0, imgarr.shape[0]-1, num_giblets)
y0 = np.random.random_integers(0, imgarr.shape[1]-1, num_giblets)

x1 = x0 + np.random.random_integers(1, max_giblet_size, num_giblets)
y1 = y0 + np.random.random_integers(1, max_giblet_size, num_giblets)

random_pool = []
rnds = np.random.random(x0.shape)

for i in xrange(len(x0)):
    if rnds[i] > .8:
        block = imgarr[x0[i]:x1[i], y0[i]:y1[i]]
        m =  block.mean(axis=1).mean(axis=0)
        random_pool.append(m)
        imgarr[x0[i]:x1[i], y0[i]:y1[i]] = rnd.choice(random_pool)
    elif rnds[i] > .5:
        block = imgarr[x0[i]:x1[i], y0[i]:y1[i]]
        m =  block.mean(axis=1).mean(axis=0)
        random_pool.append(m)
        imgarr[x0[i]:x1[i], y0[i]:y1[i]] = m
    elif rnds[i] > .4:
        line = rnd.choice(imgarr)
        imgarr[x0[i]:x1[i]] = line
    else:
        pass
    #else:
        #col = imgarr[:,rnd.choice(xrange(imgarr.shape[1]))]
        #print imgarr[:, y0[i]: y1[i]].shape

imgarr = np.rot90(imgarr).copy()
x0 = np.random.random_integers(0, imgarr.shape[0]-1, num_giblets)
y0 = np.random.random_integers(0, imgarr.shape[1]-1, num_giblets)

x1 = x0 + np.random.random_integers(1, max_giblet_size, num_giblets)
y1 = y0 + np.random.random_integers(1, max_giblet_size, num_giblets)

random_pool = []
rnds = np.random.random(x0.shape)

for i in xrange(len(x0)):
    if rnds[i] > .8:
        block = imgarr[x0[i]:x1[i], y0[i]:y1[i]]
        m =  block.mean(axis=1).mean(axis=0)
        random_pool.append(m)
        imgarr[x0[i]:x1[i], y0[i]:y1[i]] = rnd.choice(random_pool)
    elif rnds[i] > .5:
        block = imgarr[x0[i]:x1[i], y0[i]:y1[i]]
        m =  block.mean(axis=1).mean(axis=0)
        random_pool.append(m)
        imgarr[x0[i]:x1[i], y0[i]:y1[i]] = m
    elif rnds[i] > .4:
        line = rnd.choice(imgarr)
        imgarr[x0[i]:x1[i]] = line
    else:
        pass
        #col = imgarr[:,rnd.choice(xrange(imgarr.shape[1]))]
        #print imgarr[:, y0[i]: y1[i]].shape

imgarr = np.rot90(imgarr, 3).copy()

pyplot.imshow(imgarr)
pyplot.show()
#
