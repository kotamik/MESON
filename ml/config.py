# configuration file for optimization pipeline
EMX = 0.55 # max eccentricity
DUR = 2 # duration of trial per generation, in years
DTS = 300.00 # GA step time, in seconds
MEV = 5 # minimum elevation, in degrees
MPA = 100e3 # minimum periapsis altitude
APS = 9000e3 # soft limit on max apoapsis altitude

NOS = 8 # penalty for number of satellites
DOP = 300 # penalty for deorbits 
GAP = 0.5 # penalty for hours of gap in coverage
LON = 5 # penalty for longevity
CPX = 3 # penalty for complexity
COV = 60 # coverage bonus


CNS = False # true if you want notification sound when pipeline is done