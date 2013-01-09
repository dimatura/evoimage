import matplotlib.pyplot as pyplt
import numpy as np
from PIL import Image
import time
import os
import random

NUM_BLOBS = 120
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.4
MAX_ITERATIONS = 4000
POPSIZE = 80
SURVIVAL_RATE = 0.5

(X0, Y0, X1, Y1, R, G, B) = range(7)

def dump_args(func):
    """
    This decorator dumps out the arguments passed to a function before calling
    it.
    """
    argnames = func.func_code.co_varnames[ :func.func_code.co_argcount]
    fname = func.func_name
    def echo_func(*args, **kwargs):
        print fname, ":", ', '.join(
                '%s=%r' % entry
                for entry in zip(argnames,args) + kwargs.items())
        return func(*args, **kwargs)
    return echo_func

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
    """
    arr = arr.copy()
    arr.clip(0, 255, out = arr)
    im = Image.fromarray(arr.astype('uint8'))
    im.save(filename)

class Individual(object):
    def __init__(self, data = None):
        # data is (x0, y0, x1, y1, red, green, blue)
        if data is not None:
            self.data = data
        else:
            self.data = np.random.random_sample((NUM_BLOBS, 7))
            self.repair()
        self.fitness = -1

    def clone(self):
        newind = Individual(self.data.copy())
        newind.fitness = self.fitness
        return newind

    def repair(self):
        # invert coords to make a selectable numpy rectangle
        self.data[:, X0:X1 + 1].sort(axis = 1)
        self.data[:, Y0:Y1 + 1].sort(axis = 1)

def mutate1(ind):
    mutated = ind.clone()
    row = np.random.random_integers(0, ind.data.shape[0] - 1)
    col = np.random.random_integers(0, ind.data.shape[1] - 1)
    mutated.data[row, col] = np.random.random()
    mutated.repair()
    mutated.fitness = -1
    return mutated

def mutate2(ind):
    mutated = ind.clone()
    row = np.random.random_integers(0, ind.data.shape[0] - 1)
    mutated.data[row] += np.random.normal(scale = 0.3,
            size = mutated.data.shape[1])
    mutated.data[row].clip(0., 1., out = mutated.data[row])
    mutated.repair()
    mutated.fitness = -1
    return mutated

def crossover1(pa, ma):
    row = np.random.random_integers(0, pa.data.shape[0] - 1)
    newdata = np.concatenate((pa.data[:row].copy(), ma.data[row:].copy()), 0)
    newind = Individual(newdata)
    return newind

def crossover2(pa, ma):
    alpha = np.random.random()
    newdata = pa.data * alpha + ma.data * (1. - alpha)
    newind = Individual(newdata)
    newind.repair()
    return newind

def nullop(ind):
    return ind

class FitnessEvaluator(object):
    def __init__(self, target_img):
        self.target_img = target_img
        img_h, img_w = target_img.shape[0], target_img.shape[1]
        self.scaler = np.array([img_w, img_h,
            img_w, img_h, 255, 255, 255], dtype = np.int)
        self.decoded = np.zeros((img_h, img_w, 3), dtype = np.int)

    def decode_data(self, ind):
        scaled = (ind.data.copy() * self.scaler)
        self.decoded.fill(0)

        for row in scaled:
            self.decoded[row[Y0]:row[Y1], row[X0]:row[X1]] += row[R:]

        # don't clip to penalize too much whiteness
        self.decoded.clip(0, 255, out = self.decoded)
        return self.decoded

    def __call__(self, ind):
        if ind.fitness == -1:
            self.decode_data(ind)
            self.decoded -= self.target_img
            self.decoded = np.abs(self.decoded)
            delta = self.decoded.sum()
            return delta
        # use cached fitness value
        return ind.fitness


class ImgEvolver(object):

    def __init__(self, image):
        self.image = image
        self.evaluator = FitnessEvaluator(image)
        # default params
        self.popsize = POPSIZE
        self.max_iterations = MAX_ITERATIONS
        self.best_fitness_log = []
        self.mean_fitness_log = []
        self.survival_rate = SURVIVAL_RATE # fraction to keep
        self.fitness = np.zeros(POPSIZE, dtype = np.int)
        self.operators = [mutate1, mutate2, crossover1, crossover2, nullop]

    def _init(self):
        self.population = [Individual() for ind in xrange(self.popsize)]
        self.timestamp = str(int(time.time()))

        os.mkdir(self.timestamp)
        self.logfile = open('%s/best_and_mean_fitness.log' % (self.timestamp), 'w')

    def _eval_fitness(self):
        for ix, ind in enumerate(self.population):
            self.fitness[ix] = self.evaluator(ind)
        sort_ix = np.argsort(self.fitness)
        # re-sort fitness and population according to fitness
        self.fitness = self.fitness[sort_ix]
        self.population = [self.population[ix] for ix in sort_ix]
        # gather stats
        mean = np.mean(self.fitness)
        best = self.fitness[0]
        self.mean_fitness_log.append(mean)
        self.best_fitness_log.append(best)

    def _report(self, iteration):
        print 'Iter:', iteration,
        print 'Best:', self.best_fitness_log[-1],
        print 'Mean:', self.mean_fitness_log[-1]
        self.logfile.write("%f\t%f\n" %
                (self.best_fitness_log[-1], self.mean_fitness_log[-1]))
        # don't write images if no improvement
        if (len(self.best_fitness_log) > 1
                and self.best_fitness_log[-1] < self.best_fitness_log[-2]):
            best_ind = self.population[0]
            decoded = self.evaluator.decode_data(best_ind)
            save_array_as_image(decoded, '%s/%04d.jpg'%(self.timestamp, iteration))

    def _apply_operators(self):
        op = random.choice(self.operators)
        if op.func_code.co_argcount == 1:
            ind = random.choice(self.population)
            new_ind = op(ind)
        elif op.func_code.co_argcount == 2:
            (pa, ma) = random.sample(self.population, 2)
            new_ind = op(pa, ma)
        return new_ind

    def _next_generation(self):
        # simple truncation selection
        survivor_count = int(self.survival_rate * self.popsize)
        self.population = self.population[:survivor_count]
        # elite
        new_pop = [self.population[0]]
        while len(new_pop) < self.popsize:
            new_ind = self._apply_operators()
            new_pop.append(new_ind)
        # swap pops
        self.population = new_pop

    def evolve(self):
        self._init()
        self._eval_fitness()
        self._report(0)
        for iter in xrange(self.max_iterations):
            self._next_generation()
            self._eval_fitness()
            self._report(iter)

def main():
    target_img = load_image_as_array('0001cropsmall.jpg')
    evo = ImgEvolver(target_img)
    evo.evolve()

if __name__ == '__main__':
    main()
