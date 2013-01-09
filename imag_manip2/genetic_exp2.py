
import sys
import time
import os
import pdb
import numpy as np
from PIL import Image
import decode_loop
import scipy.weave as weave
import fitness as cy_fitness

NUM_BLOBS = 28
MUTATION_RATE = 0.4
CROSSOVER_RATE = 0.5
MAX_ITERATIONS = 8000
POPSIZE = 80
SURVIVAL_RATE = 0.5
(X0, Y0, X1, Y1, R, G, B) = range(7)

# image load and save
#------------------------------------------------------------------------------
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

# chromosome
#------------------------------------------------------------------------------

class Individual(object):
    def __init__(self, data = None):
        # Each data row is a rectangle, encoded as (x0, y0, x1, y1, red, green,
        # blue)
        if data is not None:
            self.data = data
        else:
            self.data = np.random.random_sample((NUM_BLOBS, 7))
            self.repair()
        # A value of -1 means 'not evaluated'
        self.fitness = -1

    def clone(self):
        newind = Individual(self.data.copy())
        newind.fitness = self.fitness
        return newind

    def repair(self):
        # invert coords to make a selectable numpy rectangle
        self.data[:, X0 : (X1 + 1)].sort(axis = 1)
        self.data[:, Y0 : (Y1 + 1)].sort(axis = 1)

# operators
#------------------------------------------------------------------------------

def mutate(ind):
    r = np.random.random()
    if r < 1./3.:
        return mutate_shape(ind)
    elif np.random.random() < 2./3.:
        return mutate_color(ind)
    return mutate_position(ind)

def mutate_shape(ind):
    mutated = ind.clone()
    row = np.random.random_integers(0, ind.data.shape[0] - 1)
    col = np.random.random_integers(0, Y1)
    mutated.data[row, col] = np.random.random()
    mutated.repair()
    mutated.fitness = -1
    return mutated

def mutate_color(ind):
    mutated = ind.clone()
    row = np.random.random_integers(0, ind.data.shape[0] - 1)
    mutated.data[row, R: ] = np.random.random(3)
    mutated.repair()
    mutated.fitness = -1
    return mutated

def mutate_position(ind):
    mutated = ind.clone()
    row = np.random.random_integers(0, ind.data.shape[0] - 1)
    new_x0, new_y0 = np.random.random(2)
    dx = new_x0 - mutated.data[row, X0]
    dy = new_y0 - mutated.data[row, Y0]
    mutated.data[row, X0] = new_x0
    mutated.data[row, Y0] = new_y0
    mutated.data[row, X1] += dx
    mutated.data[row, Y1] += dy
    mutated.data[row].clip(0., 1., out = mutated.data[row])
    mutated.repair()
    mutated.fitness = -1
    return mutated

def crossover(pa, ma):
    if np.random.random() < .5:
        return crossover1(pa, ma)
    return crossover2(pa, ma)

def crossover1(pa, ma):
    """
    Kid is made from a fraction of rectangles from each parent.
    """
    row = np.random.random_integers(0, pa.data.shape[0] - 1)
    newdata = np.r_[pa.data[:row], ma.data[row:]]
    return Individual(newdata.copy())

def crossover2(pa, ma):
    """
    Kid is a linear combination of both parents.
    """
    alpha = np.random.random()
    newdata = pa.data * alpha + ma.data * (1. - alpha)
    newind = Individual(newdata)
    newind.repair()
    return newind

# Fitness function
#------------------------------------------------------------------------------

class FitnessEvaluator(object):

    def __init__(self, target_img):
        self.target_img = target_img

        img_h, img_w = target_img.shape[0], target_img.shape[1]

        self.scaler = np.array([img_w, img_h,
            img_w, img_h, 255, 255, 255], dtype = np.int)

        self.decoded = np.empty((img_h, img_w, 3), dtype = np.int)

        self.scaled = None

    # about 111 ms
    def py_decode_data(self, ind):
        self.decoded.fill(0)
        self.scaled = (ind.data * self.scaler).astype(np.int)
        for row in self.scaled:
            self.decoded[row[Y0]:row[Y1], row[X0]:row[X1]] += row[R:]
        self.decoded.clip(0, 255, out = self.decoded)

    # fastest so far, 7 ms
    def c_decode_data(self, ind):
        self.decoded.fill(0)
        self.scaled = (ind.data * self.scaler).astype(np.int)
        decode_loop.decode_loop(self.scaled, self.decoded)
        # self.decoded.clip(0, 255, out = self.decoded)

    # hanging up for some reason
    def cy_decode_data(self, ind):
        self.decoded.fill(0)
        self.scaled = (ind.data * self.scaler).astype(np.int)
        cy_fitness.cy_decode_data(self.scaled, self.decoded)
        self.decoded.clip(0, 255, out = self.decoded)

    # second fastest about 15 ms
    def weave_decode_data(self, ind):
        self.decoded.fill(0)
        self.scaled = (ind.data * self.scaler).astype(np.int)
        code = r"""
        enum {X0, Y0, X1, Y1, R, G, B};
        int row;
        int i, j;
        int x0, x1, y0, y1, r, g, b;
        for (row = 0; row < Nscaled[0]; ++row) {
            x0 = SCALED2(row, X0);
            y0 = SCALED2(row, Y0);
            x1 = SCALED2(row, X1);
            y1 = SCALED2(row, Y1);
            r  = SCALED2(row, R );
            g  = SCALED2(row, G );
            b  = SCALED2(row, B );
            for (i = y0; i < y1; ++i) {
                for (j = x0; j < x1; ++j) {
                    DECODED3(i, j, 0) += r;
                    DECODED3(i, j, 1) += g;
                    DECODED3(i, j, 2) += b;
                }
            }
        }
        """
        decoded = self.decoded
        scaled = self.scaled
        weave.inline(code, ['decoded', 'scaled'])
        self.decoded.clip(0, 255, out = self.decoded)

    def __call__(self, ind):
        if ind.fitness == -1: # ind is 'dirty'
            self.c_decode_data(ind)
            self.decoded -= self.target_img
            return np.abs(self.decoded).sum()
        # fitness hasn't changed, use cached fitness value
        return ind.fitness

# Selectors
#------------------------------------------------------------------------------

def rank_selector(population):
    """
    Assuming population is sorted from best to worst, sample in proportion
    to rank.
    """
    # make a cdf of probability of being selected
    invrange = np.arange(len(population), 0, -1, dtype = np.double)
    cdf = np.r_[0., (invrange / invrange.sum()).cumsum()]
    while True:
        u = np.random.random()
        ix = np.searchsorted(cdf, u) - 1
        yield population[ix]

def tournament_selector(population, size = 7):
    """
    Pick <size> individuals at random and return best.  We're assuming pop is
    sorted by fitness (we sort it anyway for elitism/truncation).
    """
    while True:
        sample_ix = np.random.random_integers(0, len(population) - 1, size)
        # because of sorted-ness, best ind is in smallest ix
        yield population[sample_ix.min()]

# Evolution engine
#------------------------------------------------------------------------------

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

    def _init(self):
        self.population = [Individual() for ind in xrange(self.popsize)]
        timestamp = time.strftime('%j_%H%M_%S')
        self.logdir = 'log_' + timestamp
        os.mkdir(self.logdir)
        self.logfile = open('%s/best_and_mean_fitness.log' % (self.logdir), 'w')

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
            self.evaluator.c_decode_data(best_ind)
            save_array_as_image(self.evaluator.decoded,
                    '%s/%04d.png'%(self.logdir, iteration))

    def _repopulate(self):
        # limit selection to best fraction of pop
        survivor_count = int(self.survival_rate * self.popsize)
        self.population = self.population[:survivor_count]
        # get elite and remove from old population
        new_pop = [self.population.pop(0)]
        selector = rank_selector(self.population)
        # selector = tournament_selector(self.population)
        while len(new_pop) < self.popsize:
            pa = selector.next()
            ma = selector.next()
            if np.random.random() < CROSSOVER_RATE:
                new_ind = crossover(pa, ma)
            else:
                new_ind = pa
            if np.random.random() < MUTATION_RATE:
                new_ind = mutate(new_ind)
            new_pop.append(new_ind)
        # update population to new generation
        self.population = new_pop

    def evolve(self):
        self._init()
        self._eval_fitness()
        self._report(0)
        for iter in xrange(1, self.max_iterations):
            self._repopulate()
            self._eval_fitness()
            self._report(iter)

# main loop
#------------------------------------------------------------------------------

def main(imgfile):
    target_img = load_image_as_array(imgfile)
    evo = ImgEvolver(target_img)
    evo.evolve()

if __name__ == '__main__':
    main(sys.argv[1])
