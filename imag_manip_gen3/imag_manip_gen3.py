#!/usr/bin/env python

import sys
import time
import os
import pprint
import pdb

import numpy as np
import numpy.random as nprand

from PIL import Image

# from PyQt4.QtCore import *
# from PyQt4.QtGui import *

from commandlineapp import CommandLineApp

import fitness_ext


(X0, Y0, X1, Y1, R, G, B, A) = range(8)

# image load and save
#----------------------------

def load_image_as_array(filename):
    """
    Open filename with PIL and return it as
    numpy array.
    """
    im = Image.open(filename)
    arr = np.asarray(im)
    return arr

def save_array_as_image(arr, filename):
    """
    Save array arr as image named filename using PIL.
    Array values are expected to be in range 0, 255
    """
    arr = arr.copy().clip(0, 255).astype('uint8')
    im = Image.fromarray(arr)
    im.save(filename)

#----------------------------

class IndividualFactory(object):
    def __init__(self, num_rectangles):
        self.num_rectangles = num_rectangles

    def build(self):
        # Each data row is a rectangle, encoded as (x0, y0, x1, y1, red, green,
        # blue, alpha)
        data = nprand.random_sample((self.num_rectangles, 8))
        ind = Individual(data)
        return ind

class Individual(object):
    def __init__(self, data):
        # Each data row is a rectangle, encoded as (x0, y0, x1, y1, red, green,
        # blue, alpha)
        self.data = data
        self.cost = -1

    def copy(self):
        ind = Individual(self.data.copy())
        ind.cost = self.cost
        return ind

#----------------------------

class Evolver(object):

    def __init__(self, cost_fun, individual_factory, log_fun, options):
        self.cost_fun = cost_fun
        self.individual_factory = individual_factory
        self.log_fun = log_fun
        self.options = options
        # merge options as 'self' variables for easy access
        self.__dict__.update(options)

    def init(self):
        self.population = [self.individual_factory.build() for i in xrange(self.pop_size)]
        self.cost = [self.cost_fun(x) for x in self.population]
        argmin_cost = np.argmin(self.cost)
        self.best = self.population[argmin_cost].data.copy()
        self.best_cost = self.cost[argmin_cost]
        self.best_cost_log = [self.best_cost]
        self.mean_cost_log = [np.mean(self.cost)]
        timestamp = time.strftime('%j_%H%M_%S')
        self.logdir = 'log_' + timestamp
        os.mkdir(self.logdir)
        self.logfile = open('%s/best_and_mean_cost.log' % self.logdir, 'w')
        paramfile = open('%s/parameters.log' % self.logdir, 'w')
        paramfile.write(pprint.pformat(self.options) + '\n')
        paramfile.close()

    def evolve(self):
        self.init()
        iteration = 0
        while iteration < self.iterations:
            self.report(iteration)
            self.eval_cost()
            self.next_generation()
            iteration += 1
        self.report(iteration)

    def next_generation(self):
        # limit selection to best fraction of pop
        survivor_count = int(self.survival_rate * self.pop_size)
        self.population = self.population[:survivor_count]
        # get elite and remove from old population
        new_pop = [self.population.pop(0)]
        ind_generator = self.selector(self.population)
        # ind_generator = tournament_selector(self.population)
        while len(new_pop) < self.pop_size:
            pa = ind_generator.next()
            ma = ind_generator.next()
            if nprand.random() < self.xover_rate:
                # get it on!
                new_ind = crossover(pa, ma)
            else:
                # insert ma or pa, unmodified
                if nprand.rand() < .5:
                    new_ind = pa
                else:
                    new_ind = ma
            if nprand.random() < self.mut_rate:
                # mutate new individual
                new_ind = mutate(new_ind)

            new_pop.append(new_ind)
            #steady_state_reinserter(new_ind, self.population, new_pop)

        # update population to new generation
        self.population = new_pop

    def eval_cost(self):
        self.cost = [self.cost_fun(x) for x in self.population]
        sort_ix = np.argsort(self.cost)
        # re-sort cost and population according to cost
        self.cost = [self.cost[ix] for ix in sort_ix]
        self.population = [self.population[ix] for ix in sort_ix]
        # gather stats
        mean = np.mean(self.cost)
        self.best = self.population[0].data.copy()
        self.best_cost = self.cost[0]
        self.mean_cost_log.append(mean)
        self.best_cost_log.append(self.best_cost)

    def report(self, iteration):
        print 'Iter:', iteration,
        print 'Best:', self.best_cost_log[-1],
        print 'Mean cost:', self.mean_cost_log[-1]
        self.logfile.write("%f\t%f\n" % (self.best_cost_log[-1],
            self.mean_cost_log[-1]))
        self.log_fun(iteration, self.logdir, self.population,
                self.best, self.best_cost)

# Selectors
#----------------------------

def rank_selector(population):
    """
    Assuming population is sorted from best to worst, sample in proportion
    to rank.
    """
    # make a cdf of probability of being selected
    invrange = np.arange(len(population), 0, -1, dtype = np.double)
    cdf = np.r_[0., (invrange / invrange.sum()).cumsum()]
    while True:
        u = nprand.random()
        ix = np.searchsorted(cdf, u) - 1
        yield population[ix]

def tournament_selector(population, size = 5):
    """
    Pick <size> individuals at random and return best.  We're assuming pop is
    sorted by cost (we sort it anyway for elitism/truncation).
    """
    while True:
        sample_ix = nprand.random_integers(0, len(population) - 1, size)
        # because of sorted-ness, best ind is in smallest ix
        yield population[sample_ix.min()]

# reinsertion policies
#----------------------------

def default_reinserter(new_ind, old_pop, new_pop):
    "just appends"
    new_pop.append(new_ind)

def steady_state_reinserter(new_ind, old_pop, new_pop):
    "only appends if new_ind is better than old_pop's worst"
    if new_ind.cost < old_pop[-1].cost:
        new_pop.append(new_ind)

# operators
#----------------------------

def mutate(ind):
    return mutate_simple(ind)

def mutate_simple(ind):
    "just blast one spot"
    mutated = ind.copy()
    row = nprand.random_integers(0, ind.data.shape[0] - 1)
    col = nprand.random_integers(0, ind.data.shape[1] - 1)
    mutated.data[row, col] = nprand.random()
    mutated.cost = -1
    return mutated

def mutate_shape(ind):
    "change shape points"
    thresh = nprand.random()
    mutated = ind.copy()
    rows = nprand.random(len(ind.data))
    to_mutate = mutated.data[rows < thresh , : Y1 + 1]
    to_mutate = np.random.random(to_mutate.shape)
    mutated.data[rows < thresh, : Y1 + 1] = to_mutate
    mutated.cost = -1
    return mutated

def mutate_color(ind):
    "change color and alpha"
    thresh = nprand.random()
    mutated = ind.copy()
    rows = nprand.random(len(ind.data))
    to_mutate = mutated.data[rows < thresh , Y1:]
    to_mutate = np.random.random(to_mutate.shape)
    mutated.data[rows < thresh, Y1 :] = to_mutate
    mutated.cost = -1
    return mutated

def mutate_reorder(ind):
    "reorder rectangles "
    mutated = ind.copy()
    for i in xrange(3):
        row1 = nprand.random_integers(0, ind.data.shape[0] - 1)
        row2 = nprand.random_integers(0, ind.data.shape[0] - 1)
        mutated.data[row1], mutated.data[row2] =\
                mutated.data[row2].copy(), mutated.data[row1].copy()
    mutated.cost = -1
    return mutated

def mutate_position(ind):
    "change rect position"
    thresh = nprand.random()
    scale = nprand.uniform(0.01, 0.5)
    mutated = ind.copy()
    rows = nprand.random(len(ind.data))
    to_mutate = mutated.data[rows < thresh]
    dx = nprand.normal(0, scale, to_mutate.shape[0])
    dy = nprand.normal(0, scale, to_mutate.shape[0])
    to_mutate[:, X0] += dx
    to_mutate[:, Y0] += dy
    to_mutate[:, X1] += dx
    to_mutate[:, Y1] += dy
    to_mutate.clip(0., 1., out=to_mutate)
    mutated.data[rows < thresh] = to_mutate
    mutated.cost = -1
    return mutated

def mutate_noise(ind):
    "add noise to whole individual"
    mutated = ind.copy()
    scale = nprand.uniform(0.01, 0.1)
    mutated.data += nprand.normal(0., scale, mutated.data.shape)
    mutated.data.clip(0., 1., out = mutated.data)
    mutated.cost = -1
    return mutated


def crossover(pa, ma):
    return crossover_unif(pa, ma)
    # i'm not very happy with arithmetic xover
    # return crossover_arith(pa, ma)

def crossover_unif(pa, ma):
    """
    Kid is made from a fraction of rectangles from each parent.
    Uniform crossover.
    """
    mask = nprand.random_integers(0, 1, len(pa.data))
    kiddata = np.array([pa_r if mask_r else ma_r for (pa_r, ma_r, mask_r) in
        zip(pa.data, ma.data, mask)])
    return Individual(kiddata.copy())

def crossover_arith(pa, ma):
    """
    Kid is a linear combination of both parents.
    """
    alpha = nprand.uniform(0, 2.)
    newdata = np.empty(pa.data.shape)
    if alpha < 1.0:
        newdata[:, R:] = pa.data[:, R:].copy()
    else:
        newdata[:, R:] = ma.data[:, R:].copy()
    newdata[:, :R]  = alpha * pa.data[:, :R]+ (1 - alpha) * ma.data[:, :R]
    newdata.clip(0., 1., out = newdata)
    newind = Individual(newdata)
    return newind

#----------------------------

# class ImageMontageForm(QDialog):
#     def __init__(self, parent = None)
#         super(Form, self).__init__(parent)
#         grid = QGridLayout()
#         # grid.addWidget(foo, 0, 0)
#         self.setLayout(grid)
#         self.setWindowTitle("Population")
#
#     def __call__(self, iteration, logdir,
#             population, best, best_cost):
#         pass
#
#     def getdata(self):
#         pass
#
#     def updateUi(self):
#         pass

class ImageLog(object):
    """ application specific logger, writes img """
    def __init__(self, img_shape):
        self.img_shape = img_shape
        self.best_so_far = -1

    def __call__(self, iteration, logdir,
            population, best, best_cost):
        # don't write images if no improvement
        if (self.best_so_far != -1 and
                best_cost >= self.best_so_far):
            return
        self.best_so_far = best_cost
        #-TODO eliminate dependence on logdir
        img_h, img_w = self.img_shape[:2]
        decoded = np.empty((img_h, img_w, 3), dtype = np.int)
        fitness_ext.decode_rectangles(best, decoded)
        save_array_as_image(decoded, '%s/%04d.png'%(logdir, iteration))

#----------------------------

class ImageDiffCost(object):
    def __init__(self, target_img):
        self.target_img = target_img
        img_h, img_w = target_img.shape[0], target_img.shape[1]
        self.decoded = np.empty((img_h, img_w, 3), dtype = np.int)

    def decode(self, individual):
        fitness_ext.decode_rectangles(individual.data, self.decoded)
        #self.decoded.clip(0, 255, out = self.decoded)
        #self.decoded = 255 - self.decoded

    def __call__(self, individual):
        # test for 'cached' value
        if individual.cost != -1:
            return individual.cost
        self.decode(individual)
        d = fitness_ext.img_diff(self.target_img, self.decoded)
        individual.cost = d
        return d


# main
#----------------------------
class ImagEvolve(CommandLineApp):
    """
    Build images with rectangles and PSO.
    """
    def before_options_hook(self):
        self.options = {
                'iterations' : 8000,
                'pop_size' : 50,
                'num_rectangles' : 28,
                'xover_rate' : .7,
                'mut_rate' : .2,
                'survival_rate' : .5,
                'selector' : rank_selector,
                }

    def option_handler_iterations(self, num):
        "Set maximum number of iterations"
        self.options['iterations'] = int(num)

    def option_handler_pop_size(self, num):
        "Set number of individuals (population size)"
        self.options['pop_size'] = int(num)

    def option_handler_num_rectangles(self, num):
        "Set number of rectangles"
        self.options['num_rectangles'] = int(num)

    def option_handler_mut_rate(self, num):
        "Set mutation rate"
        self.options['mut_rate'] = float(num)

    def option_handler_xover_rate(self, num):
        "Set xrossover rate"
        self.options['xover_rate'] = float(num)

    def option_handler_survival_rate(self, num):
        "Set survival rate"
        self.options['survival_rate'] = float(num)

    def option_handler_selector(self, name):
        "Set selection algorithm"
        try:
            selector = {
                    'rank' : rank_selector,
                    'tournament' : tournament_selector,
                    }[name]
            self.options['selector'] = selector
        except KeyError:
            self.error_message("Invalid selector")
            return 1

    def main(self, *args):
        if len(args) == 0:
            self.error_message("Input image required")
            return 1
        elif len(args) > 1:
            self.error_message("Only one image allowed")
            return 1
        try:
            target_img = load_image_as_array(args[0])
        except Exception, ex:
            self.error_message("Couldn't read image: %r" % ex)
            return 1
        self.status_message("Using options: %s" % pprint.pformat(self.options))
        cost_fun = ImageDiffCost(target_img)
        individual_factory = IndividualFactory(self.options['num_rectangles'])

        # report/log fun is a bit more complex because it has gui
        save_img_log_fun = ImageLog(target_img.shape)
        # gui_log_fun = ImageMontageForm()
        def img_log_fun(*args):
            save_img_log_fun(*args)
        #    gui_log_fun(*args)

        # this will have to be in its own thread I think
        # app = QApplication(sys.argv)
        # gui_log_fun.show()
        # app.exec_()

        evolver = Evolver(cost_fun, individual_factory, img_log_fun, self.options)
        evolver.evolve()

if __name__ == '__main__':
    ImagEvolve().run()
