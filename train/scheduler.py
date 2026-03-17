import math

def scheduler(epoch, restart_period=70, restart_factor=2):
    T = restart_period
    # while epoch >= T:
    #     epoch -= T
    #     T *= restart_factor
    return 0.5 * (1 + math.cos(math.pi * epoch / T))

