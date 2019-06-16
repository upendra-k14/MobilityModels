"""
Code for RPGM model was used from Github repo:
https://github.com/panisson/pymobility
"""
import numpy as np
from numpy.random import rand

# define a Uniform Distribution
U = lambda MIN, MAX, SAMPLES: rand(*SAMPLES.shape) * (MAX - MIN) + MIN

def reference_point_group(nr_nodes, dimensions, velocity=(0.1, 1.), aggregation=0.1):
    '''
    Reference Point Group Mobility model, discussed in the following paper:

        Xiaoyan Hong, Mario Gerla, Guangyu Pei, and Ching-Chuan Chiang. 1999.
        A group mobility model for ad hoc wireless networks. In Proceedings of the
        2nd ACM international workshop on Modeling, analysis and simulation of
        wireless and mobile systems (MSWiM '99). ACM, New York, NY, USA, 53-60.

    In this implementation, group trajectories follow a random direction model,
    while nodes follow a random walk around the group center.
    The parameter 'aggregation' controls how close the nodes are to the group center.

    Required arguments:

      *nr_nodes*:
        list of integers, the number of nodes in each group.

      *dimensions*:
        Tuple of Integers, the x and y dimensions of the simulation area.

    keyword arguments:

      *velocity*:
        Tuple of Doubles, the minimum and maximum values for group velocity.

      *aggregation*:
        Double, parameter (between 0 and 1) used to aggregate the nodes in the group.
        Usually between 0 and 1, the more this value approximates to 1,
        the nodes will be more aggregated and closer to the group center.
        With a value of 0, the nodes are randomly distributed in the simulation area.
        With a value of 1, the nodes are close to the group center.
    '''

    try:
        iter(nr_nodes)
    except TypeError:
        nr_nodes = [nr_nodes]

    NODES = np.arange(sum(nr_nodes))

    groups = []
    prev = 0
    for (i,n) in enumerate(nr_nodes):
        groups.append(np.arange(prev,n+prev))
        prev += n

    g_ref = np.empty(sum(nr_nodes), dtype=np.int)
    for (i,g) in enumerate(groups):
        for n in g:
            g_ref[n] = i

    FL_MAX = max(dimensions)
    MIN_V,MAX_V = velocity
    FL_DISTR = lambda SAMPLES: U(0, FL_MAX, SAMPLES)
    VELOCITY_DISTR = lambda FD: U(MIN_V, MAX_V, FD)

    MAX_X, MAX_Y = dimensions
    x = U(0, MAX_X, NODES)
    y = U(0, MAX_Y, NODES)
    velocity = 1.
    theta = U(0, 2*np.pi, NODES)
    costheta = np.cos(theta)
    sintheta = np.sin(theta)

    GROUPS = np.arange(len(groups))
    g_x = U(0, MAX_X, GROUPS)
    g_y = U(0, MAX_X, GROUPS)
    g_fl = FL_DISTR(GROUPS)
    g_velocity = VELOCITY_DISTR(g_fl)
    g_theta = U(0, 2*np.pi, GROUPS)
    g_costheta = np.cos(g_theta)
    g_sintheta = np.sin(g_theta)

    while True:

        x = x + velocity * costheta
        y = y + velocity * sintheta

        g_x = g_x + g_velocity * g_costheta
        g_y = g_y + g_velocity * g_sintheta

        for (i,g) in enumerate(groups):

            # step to group direction + step to group center
            x_g = x[g]
            y_g = y[g]
            c_theta = np.arctan2(g_y[i] - y_g, g_x[i] - x_g)

            x[g] = x_g + g_velocity[i] * g_costheta[i] + aggregation*np.cos(c_theta)
            y[g] = y_g + g_velocity[i] * g_sintheta[i] + aggregation*np.sin(c_theta)

        # node and group bounces on the margins
        b = np.where(x<0)[0]
        if b.size > 0:
            x[b] = - x[b]; costheta[b] = -costheta[b]
            g_idx = np.unique(g_ref[b]); g_costheta[g_idx] = -g_costheta[g_idx]
        b = np.where(x>MAX_X)[0]
        if b.size > 0:
            x[b] = 2*MAX_X - x[b]; costheta[b] = -costheta[b]
            g_idx = np.unique(g_ref[b]); g_costheta[g_idx] = -g_costheta[g_idx]
        b = np.where(y<0)[0]
        if b.size > 0:
            y[b] = - y[b]; sintheta[b] = -sintheta[b]
            g_idx = np.unique(g_ref[b]); g_sintheta[g_idx] = -g_sintheta[g_idx]
        b = np.where(y>MAX_Y)[0]
        if b.size > 0:
            y[b] = 2*MAX_Y - y[b]; sintheta[b] = -sintheta[b]
            g_idx = np.unique(g_ref[b]); g_sintheta[g_idx] = -g_sintheta[g_idx]

        # update info for nodes
        theta = U(0, 2*np.pi, NODES)
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # update info for arrived groups
        g_fl = g_fl - g_velocity
        g_arrived = np.where(np.logical_and(g_velocity>0., g_fl<=0.))[0]

        if g_arrived.size > 0:
            g_theta = U(0, 2*np.pi, g_arrived)
            g_costheta[g_arrived] = np.cos(g_theta)
            g_sintheta[g_arrived] = np.sin(g_theta)
            g_fl[g_arrived] = FL_DISTR(g_arrived)
            g_velocity[g_arrived] = VELOCITY_DISTR(g_fl[g_arrived])

        yield np.dstack((x,y))[0]
