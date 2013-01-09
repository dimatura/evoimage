
import numpy as np
import matplotlib.pyplot as pyplt
import genetic_exp2 as g
import pdb

g.NUM_BLOBS = 4
ROWS = 5
COLS = 4
ITERS = ROWS * COLS

def test_mutate():
    fiteval = g.FitnessEvaluator(np.zeros((240, 320)))
    ind = g.Individual()

    for ix in xrange(ITERS):
        pyplt.subplot(ROWS, COLS, ix + 1)
        fiteval.cy_decode_data(ind)
        dec = fiteval.decoded.copy().astype('uint8')
        pyplt.axis('off')
        pyplt.imshow(dec)
        ind = g.mutate(ind)

    pyplt.show()

def test_xover():
    fiteval = g.FitnessEvaluator(np.zeros((240, 320)))
    pa = g.Individual()
    ma = g.Individual()

    for ix, ind in enumerate((pa, ma)):
        pyplt.subplot(ROWS, COLS, ix + 2)
        fiteval.cy_decode_data(ind)
        dec = fiteval.decoded.copy().astype('uint8')
        pyplt.axis('off')
        pyplt.imshow(dec)

    for ix in xrange(4, ITERS):
        ind = g.crossover2(pa, ma)
        pyplt.subplot(ROWS, COLS, ix + 1)
        fiteval.cy_decode_data(ind)
        dec = fiteval.decoded.copy().astype('uint8')
        pyplt.axis('off')
        pyplt.imshow(dec)

    pyplt.show()


test_mutate()
