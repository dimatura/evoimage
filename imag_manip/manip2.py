from PIL import Image
import matplotlib.pyplot as pyplot
import numpy as np
import random as rnd

img = Image.open('0001resz.jpg')
imgarr = np.asarray(img).copy()

num_giblets = 8
#max_giblet_size = 3
#max_giblet_size = (30,30)

x_splits = np.random.random_integers(0, imgarr.shape[0]-1, num_giblets)
x_splits.sort()

rows = np.array_split(imgarr, x_splits)
rows = [r for r in rows if r.size > 0]
rnd.shuffle(rows)

new_rows = []
for row in rows:
    num_giblets = np.random.random_integers(1,200)
    y_splits = np.random.random_integers(0, imgarr.shape[1]-1, num_giblets)
    y_splits.sort()
    cols = np.array_split(row, y_splits, axis = 1)
    cols = [c for c in cols if c.size > 0]
    rnd.shuffle(cols)
    row = np.concatenate(cols, axis = 1)
    new_rows.append(row)

imgarr = np.concatenate(new_rows, axis = 0)

#y_splits = np.random.random_integers(0, imgarr.shape[1]-1, num_giblets)
#y_splits.sort()

pyplot.imshow(imgarr)
pyplot.show()
#
