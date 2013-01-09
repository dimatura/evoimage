#!/usr/bin/env python

import time
import os
import pprint

import numpy as np
from PIL import Image

from commandlineapp import CommandLineApp

import fitness_ext

def py_decode_rectangles(scaled, decoded):
    (X0, Y0, X1, Y1, R, G, B) = range(7)
    decoded.fill(0)
    for row in scaled:
        decoded[row[Y0]:row[Y1], row[X0]:row[X1]] += row[R:]

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

class ParticleBuilder(object):
    def __init__(self, num_rectangles, max_velocity):
        self.num_rectangles = num_rectangles
        self.max_velocity = max_velocity

    def build(self):
        # Each data row is a rectangle, encoded as (x0, y0, x1, y1, red, green,
        # blue, alpha)
        position = np.random.random_sample((self.num_rectangles, 8))
        velocity = np.random.uniform(-self.max_velocity, self.max_velocity,
                (self.num_rectangles, 8))
        personal_best = position.copy()
        return Particle(position, velocity, personal_best)

class Particle(object):
    def __init__(self, position, velocity, personal_best):
        # Each data row is a rectangle, encoded as (x0, y0, x1, y1, red, green,
        # blue, alpha)
        self.position = position
        self.velocity = velocity
        self.personal_best = personal_best

#----------------------------
class Swarmer(object):
    def __init__(self, cost_fun, particle_builder, log_fun, options):
        self.cost_fun = cost_fun
        self.particle_builder = particle_builder
        self.log_fun = log_fun
        self.options = options
        # merge options as 'self' variables for easy access
        self.__dict__.update(options)

    def init(self):
        self.population = [self.particle_builder.build() for i in xrange(self.pop_size)]
        self.cost = [self.cost_fun(x) for x in self.population]
        argmin_cost = np.argmin(self.cost)
        self.global_best = self.population[argmin_cost].position.copy()
        self.global_best_cost = self.cost[argmin_cost]
        self.mean_vel = np.mean([np.abs(p.velocity) for p in self.population])

        self.best_cost_log = [self.global_best_cost]
        self.mean_cost_log = [np.mean(self.cost)]
        timestamp = time.strftime('%j_%H%M_%S')
        self.logdir = 'log_' + timestamp
        os.mkdir(self.logdir)
        self.logfile = open('%s/best_and_mean_fitness.log' % (self.logdir), 'w')
        paramfile = open('%s/parameters.log' % self.logdir, 'w')
        paramfile.write(pprint.pformat(self.options) + '\n')
        paramfile.close()

    def swarm(self):
        self.init()
        iteration = 0
        while iteration < self.iterations:
            self.report(iteration)
            self.eval_cost()
            self.update_particles()
            iteration += 1
        self.report(iteration)

    def update_particles(self):
        """
        Standard PSO '07 docs:
        For each particle and each dimension
        Equation 1:	v(t+1) = w*v(t) + R(c)*(p(t)-x(t)) + R(c)*(g(t)-x(t))
        Equation 2:	x(t+1) = x(t) + v(t+1)
        where
        w := first cognitive/confidence coefficient
        c := second cognitive/confidence coefficient
        v(t) := velocity at time t
        x(t) := position at time t
        p(t) := best previous position of the particle
        g(t) := best position amongst the best previous positions
                        of the informants of the particle
        R(c) := a number coming from a random distribution, which depends on c
        In this standard, the distribution is uniform on [0,c]
        """
	# According to Clerc's Stagnation Analysis
	w = 0.721
	c = 1.193
        for ix, p in enumerate(self.population):
            r1 = np.random.uniform(0., c, p.position.shape)
            r2 = np.random.uniform(0., c, p.position.shape)
            p.velocity *= w
            p.velocity += r1 * (p.personal_best - p.position)
            p.velocity += r2 * (self.global_best - p.position)
            # limit max velocity
            p.velocity.clip(-self.max_velocity, self.max_velocity, out = p.velocity)
            overlimit = p.position > 1.
            underlimit = p.position < 0.
            # reduce velocity of out of bounds particles
            p.velocity[overlimit] = 0.
            p.velocity[underlimit] = 0.
            # update position
            p.position += p.velocity
            # limit final position
            p.position.clip(0., 1., out = p.position)

    def eval_cost(self):
        old_cost = self.cost[:]
        self.cost = [self.cost_fun(x) for x in self.population]
        argmin_cost = np.argmin(self.cost)

        if self.cost[argmin_cost] < self.global_best_cost:
            self.global_best = self.population[argmin_cost].position.copy()
            self.global_best_cost = self.cost[argmin_cost]

        for ix, p in enumerate(self.population):
            for n_ix in xrange(ix - self.num_neighbors,
                    ix + self.num_neighbors + 1):
                n_ix %= len(self.population)
            # choose neighbors at random
            #for n_ix in np.random.random_integers(0, self.pop_size - 1,
            #        self.num_neighbors).tolist() + ix:
                if self.cost[n_ix] < old_cost[ix]:
                    p.personal_best = self.population[n_ix].position.copy()

        # gather stats
        mean = np.mean(self.cost)
        best = self.global_best_cost
        self.mean_cost_log.append(mean)
        self.best_cost_log.append(best)
        self.mean_vel = np.mean([np.abs(p.velocity) for p in self.population])

    def report(self, iteration):
        print 'Iter:', iteration,
        print 'Best:', self.best_cost_log[-1],
        print 'Mean cost:', self.mean_cost_log[-1],
        print 'Mean vel:', self.mean_vel
        self.logfile.write("%f\t%f\t%f\n" %
                (self.best_cost_log[-1],
                 self.mean_cost_log[-1],
                 self.mean_vel))
        self.log_fun(iteration, self.logdir, self.population,
                self.global_best, self.global_best_cost)

#----------------------------

class ImageLog(object):
    """ application specific logger, writes img """
    def __init__(self, img_shape):
        self.img_shape = img_shape
        self.best_so_far = -1

    def __call__(self, iteration, logdir,
            population, global_best, global_best_cost):
        # don't write images if no improvement
        if (self.best_so_far != -1 and
                global_best_cost >= self.best_so_far):
            return
        self.best_so_far = global_best_cost
        #-TODO eliminate dependence on logdir
        img_h, img_w = self.img_shape[:2]
        decoded = np.empty((img_h, img_w, 3), dtype = np.int)
        fitness_ext.decode_rectangles(global_best, decoded)
        save_array_as_image(decoded, '%s/%04d.png'%(logdir, iteration))

#----------------------------

class ImageDiffCost(object):
    def __init__(self, target_img):
        self.target_img = target_img
        img_h, img_w = target_img.shape[0], target_img.shape[1]
        self.decoded = np.empty((img_h, img_w, 3), dtype = np.int)

    def decode(self, particle):
        fitness_ext.decode_rectangles(particle.position, self.decoded)
        #self.decoded.clip(0, 255, out = self.decoded)
        #self.decoded = 255 - self.decoded

    def __call__(self, particle):
        self.decode(particle)
        d = fitness_ext.img_diff(self.target_img, self.decoded)
        return d

# main
#----------------------------
class ImagMakePso(CommandLineApp):
    """
    Build images with rectangles and PSO.
    """
    def before_options_hook(self):
        self.options = {'iterations' : 1000,
                'pop_size' : 50,
                'num_rectangles' : 28,
                'max_velocity' : 0.05,
                'num_neighbors' : 3}

    def option_handler_iterations(self, num):
        "Set maximum number of iterations"
        self.options['iterations'] = int(num)

    def option_handler_pop_size(self, num):
        "Set number of particles (population size)"
        self.options['pop_size'] = int(num)

    def option_handler_num_rectangles(self, num):
        "Set number of rectangles"
        self.options['num_rectangles'] = int(num)

    def option_handler_max_velocity(self, num):
        "Set maximum velocity"
        self.options['max_velocity'] = float(num)

    def option_handler_num_neighbors(self, num):
        "Set number of neighbors"
        self.options['num_neighbors'] = int(num)

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
        particle_builder = ParticleBuilder(self.options['num_rectangles'],
                self.options['max_velocity'])
        img_log_fun = ImageLog(target_img.shape)
        pso = Swarmer(cost_fun, particle_builder, img_log_fun, self.options)
        pso.swarm()

if __name__ == '__main__':
    ImagMakePso().run()
