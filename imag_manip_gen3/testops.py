#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as pyplot

import imag_manip_gen3 as g
import fitness_ext as f

IMG_SHAPE = (240, 300, 3)

indfactory = g.IndividualFactory(120)

def draw_in_grid(img, shape, row, col):
    pyplot.subplot(shape[0], shape[1], shape[1]*row + col + 1)
    pyplot.axis("off")
    pyplot.imshow(img.copy().astype('uint8'))

def test_mutate():
    mutators = [g.mutate_shape, g.mutate_color, g.mutate_reorder,
            g.mutate_position, g.mutate_noise]

    shape = (len(mutators) + 1, 4)

    ind = indfactory.build()
    decoded = np.empty(IMG_SHAPE, dtype = np.int)
    f.decode_rectangles(ind.data, decoded)
    orig_decoded = decoded.copy()
    draw_in_grid(decoded, shape, 0, 0)

    for row in xrange(1, shape[0]):
        for col in xrange(shape[1]):
            mutated = mutators[row - 1](ind)
            # mutated = g.mutate(ind)
            f.decode_rectangles(mutated.data, decoded)
            draw_in_grid(decoded, shape, row, col)

def test_xover():
    xovers = [g.crossover_unif, g.crossover_arith]
    pa = indfactory.build()
    ma = indfactory.build()
    decoded = np.empty(IMG_SHAPE, dtype = np.int)
    shape = (len(xovers) + 1, 10)
    f.decode_rectangles(pa.data, decoded)
    draw_in_grid(decoded, shape, 0, 0)
    f.decode_rectangles(ma.data, decoded)
    draw_in_grid(decoded, shape, 0, 1)
    for row in xrange(1, shape[0]):
        for col in xrange(shape[1]):
            kid = xovers[row - 1](pa, ma)
            f.decode_rectangles(kid.data, decoded)
            draw_in_grid(decoded, shape, row, col)

# test_mutate()
test_xover()
pyplot.show()
