import numpy as np

elements = [6.5, 4.54, 7.1, 6.6, 7.2, 8.3, 7.6, 7.9, 8.5, 9.3]
mean = np.mean(elements)
stddev = np.std(elements)

#amp = 5
weights = [((mean-e)/stddev) for e in elements]
print "Weights: {0}".format(weights)
i = 10
new_elements = []

center_point = 55
for w in weights:
  center_point += (np.absolute((55-i))*w)
  #new_elements.append(w*i)
  i += 10
#print new_elements


#center_point = np.absolute(np.mean(new_elements))
print center_point

#Our steps are as follows
#Use the average run for each configuration
#Get last n points for each run, n = 10
  #2d array
#avg array = average for each array of points in the big array
#get polyfit of data points with degree 2
#center point =  minimum of polyfit function
#step *= .1
#min = round_up(center_point, step)
#max = round_up(center_point, step)

#range = [min, max]
#step = step
#run configs again
#repeat
#repeat
#repeat
#repeat
#repeat
#repeat
#repeat
#repeat
#repeat
#repeat
#repeat

#For this to actually get automated with n variables it involves finding the minimum of a 
#n - dimensional function, which is just MORE FUCKING MACHINE LEARNING GOD DAMMIT
