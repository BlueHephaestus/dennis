import librosa
import numpy as np
import os
import cPickle, gzip #for storing our data 
import theano
import theano.tensor as T
#Get our big array of data and seperate it into training, validation, and test data
#According to the percentages passed in

##Defaults:
#80% - Training
#10% - Validation
#10% - Test
def unison_shuffle(a, b):
  rng_state = np.random.get_state()
  np.random.shuffle(a)
  np.random.set_state(rng_state)
  np.random.shuffle(b)

def get_data_subsets(archive_dir="../data/mfcc_samples.pkl.gz", p_training=0.8, p_validation=0.1, p_test=0.1):
  print "Getting Training, Validation, and Test Data..."
  
  f = gzip.open(archive_dir, 'rb')
  data = cPickle.load(f)

  """
  The following is specific to LIRA implementation:
    We have our X as shape (95*64, 80*145), and 
    our Y as shape (95*64,)

    This is because we have 95 full images, and since we 
    divide them into 80x145 subsections, we end up with 
    64 subsections for each image, hence 95*64.

  However, Professor Asa Ben-Hur raised a good point in that
    if I am just mixing them up like this, then I could have one of my 
    test data samples(a subsection, remember) be a subsection right 
    next to a training data sample, quite easily. This would ruin
    the validity of our validation and test data, so here's the plan to
    fix that:
 
  We instead split our (95*64, 80*145) matrix into 95 parts, so that we end up 
    with a 3d array of shape (95, 64, 80*145). (Do the same for Y but with a 
    resulting 2d array from the vector) Then, we can do our unison shuffle on the 
    entire images instead of our massive subsection collage.

  This way, we have entire different images for test and validation data than our
    training data, and we ensure valid results when testing.

  With that said,
  BEGIN LIRA SPECIFIC IMAGE STUFF
  """

  n_samples = 95
  data[0] = np.split(data[0], n_samples)
  data[1] = np.split(data[1], n_samples)
  "END LIRA STUFF FOR NOW"

  "USUAL STUFF"
  #n_samples = len(data[0])
  "END USUAL STUFF"

  training_data = [[], []]
  validation_data = [[], []]
  test_data = [[], []]

  n_training_subset = np.floor(p_training*n_samples)
  n_validation_subset = np.floor(p_validation*n_samples)
  #Assign this to it's respective percentage and whatever is left
  n_test_subset = n_samples - n_training_subset - n_validation_subset

  #Shuffle while retaining element correspondence
  print "Shuffling data..."
  unison_shuffle(data[0], data[1])

  #Get actual subsets
  data_x_subsets = np.split(data[0], [n_training_subset, n_training_subset+n_validation_subset])#basically the lines we cut to get our 3 subsections
  data_y_subsets = np.split(data[1], [n_training_subset, n_training_subset+n_validation_subset])

  training_data[0] = data_x_subsets[0]
  validation_data[0] = data_x_subsets[1]
  test_data[0] = data_x_subsets[2]

  training_data[1] = data_y_subsets[0]
  validation_data[1] = data_y_subsets[1]
  test_data[1] = data_y_subsets[2]
  
  "MORE LIRA SPECIFIC STUFF"
  #Now that we've shuffled and split relative to images(instead of subsections), collapse back to matrix (or vector if Y)
  #We do -1 so that it infers we want to combine the first two dimensions, and we have the last argument because we
  #want it to keep the same last dimension. repeat this for all of the subsets
  #Since Y's are just vectors, we can easily just flatten
  training_data[0] = training_data[0].reshape(-1, training_data[0].shape[-1])
  training_data[1] = training_data[1].flatten()
  validation_data[0] = validation_data[0].reshape(-1, validation_data[0].shape[-1])
  validation_data[1] = validation_data[1].flatten()
  test_data[0] = test_data[0].reshape(-1, test_data[0].shape[-1])
  test_data[1] = test_data[1].flatten()

  #print training_data[0].shape, training_data[1].shape, validation_data[0].shape, validation_data[1].shape, test_data[0].shape, test_data[1].shape
  print "# of Samples per subset:"
  print "\t{}".format(training_data[0].shape[0]/64)
  print "\t{}".format(validation_data[0].shape[0]/64)
  print "\t{}".format(test_data[0].shape[0]/64)

  print "Check to make sure these have all the different classes"
  print validation_data[1]
  print list(test_data[1])
  "END MORE LIRA SPECIFIC STUFF"

  return training_data, validation_data, test_data
