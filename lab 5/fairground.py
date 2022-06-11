# Copyright Tom SF Haines 2021

import json
import numpy



# Read in rides...
with open('rides.json') as fin:
  rides = json.load(fin)



def generate_fairground(rng = None, fullset = False):
  """Generates a random fairground, returning a tuple with two items in. First is a data matrix indexed [ride, ride feature], while second is a vector indexed [ride] that contains that rides index in the rides list.)."""
  rng = numpy.random.default_rng(rng)
  
  # Generate the random indices of the rides to include in this fairground...
  indices = numpy.arange(len(rides))
  if not fullset:
    rng.shuffle(indices)
    indices = indices[:indices.shape[0]//2]
  
  # Fill out a data matrix...
  dm = numpy.empty((indices.shape[0], 10))
  for dest, source in enumerate(indices):
    dm[dest,:] = rides[source]['fv']
  
  return dm, indices



def print_names(indices, indent = 0):
  """Given the second part of the tuple returned by generate_fairground() prints out the names of the rides selected."""
  for i in indices:
    print(' '*indent + rides[i]['name'])



def patreons(count, rng = None):
  """Returns a data matrix of patreons, indexed [patreon, patreon feature]. Only parameter is how many to generate."""
  rng = numpy.random.default_rng(rng)
  
  dm = rng.beta(2, 2, size=(count, 10))
  dm -= dm.mean(axis=1, keepdims=True)
  dm /= numpy.fabs(dm).max(axis=1, keepdims=True)
  
  return dm



def happy(assignment, rng = None):
  """Tells you if a ride makes a patreon happy. An assignment is a patreons feature vector followed by a rides feature vector. Returns 0 for unhappy and 1 for happy; includes noise so it can change with each call. Fully vectorised, so you can hand it a single vector or a 2D array of assignments. It's ultimately just a mad function, to make sure a neural network will have a challenge."""
  assert(assignment.shape[-1]==20)
  rng = numpy.random.default_rng(rng)
  
  # This is not meant to be understood. It's also not the only version...
  c0 = assignment[...,12] + assignment[...,17] - assignment[...,11] - assignment[...,15]
  c1 = assignment[...,17] + assignment[...,19] + 2*assignment[...,15] - 3*assignment[...,13] - assignment[...,16]
  c2 = assignment[...,10] + assignment[...,11] + assignment[...,12] + assignment[...,13] + assignment[...,14] + assignment[...,18] - 3
  c3 = (assignment[...,10] + assignment[...,11]) / (1 + assignment[...,12]) - 1
  c4 = (1 - assignment[...,10]) * (assignment[...,15] + assignment[...,18] - assignment[...,13] - assignment[...,16])
  c5 = numpy.square(assignment[...,13]) + numpy.sqrt(assignment[...,16]) + assignment[...,19] - 1
  c6 = numpy.power(0.25 * (assignment[...,11] + assignment[...,19] + 2*assignment[...,18] - 2*assignment[...,12] - 2*assignment[...,15]), 3)
  c7 = 2*numpy.sqrt(assignment[...,14]) - 1
  c8 = 1 - 2*numpy.square(assignment[...,15])
  
  def h1(a, c):
    delta = 2.5 * numpy.maximum(numpy.square(a), 0.25) * (0.45 - numpy.fabs(a - c))
    return delta / numpy.sqrt(1 + numpy.square(delta))
  
  h = h1(assignment[...,0], c0) + h1(assignment[...,1], c1) + h1(assignment[...,2], c2) + h1(assignment[...,3], c3) + h1(assignment[...,4], c4) + h1(assignment[...,5], c5) + h1(assignment[...,6], c6) + h1(assignment[...,7], c7) + h1(assignment[...,8], c8)
  h += 1.5 + 0.1*rng.standard_normal(size=h.shape) - assignment[...,9]
  
  return numpy.array(h>0.0, dtype=int)



if __name__=='__main__':
  import sys
  
  # Some test code to generate a fairground and a population then print out the happiness statistics, to check they are reasonable...
  rng = numpy.random.default_rng()
  
  # Generate and report on a fairground...
  fairground, indices = generate_fairground(rng, 'all' in sys.argv)
  print_names(indices)
  print()
  
  # Some paterons and an array to count how many rides make them happy...
  pop = patreons(256*1024, rng)
  pop_likes = numpy.zeros(pop.shape[0], dtype=int)
  
  # Measure happiness...
  assignment = numpy.empty((pop.shape[0], 20))
  assignment[:,:10] = pop
  
  ride_likes = numpy.empty(fairground.shape[0])
  for i, ride in enumerate(fairground):
    assignment[:,10:] = ride[None,:]
    h = happy(assignment, rng)
    pop_likes += h
    ride_likes[i] = h.sum()
 
  
  # Collate and report statistics...
  counts = numpy.bincount(pop_likes)
  print('Population like distribution:')
  for i, c in enumerate(counts):
    print('  {: 2d} likes = {:.2f}%'.format(i, 100 * c / pop.shape[0]))
  print()
  
  print('Population like percentage per ride:')
  for i, fi in enumerate(indices):
    print('  {:>18}: {:.2f}%'.format(rides[fi]['name'], 100*ride_likes[i] / pop.shape[0]))
