"""
Implementation for research paper :

Zhang, Yan, Jim Mee Ng, and Chor Ping Low. "A distributed group mobility
adaptive clustering algorithm for mobile ad hoc networks."
Computer Communications 32.1 (2009): 189-202.
"""
import numpy as np
import time

# number of nodes
N_NODES = 40

# simulation area (units)
MAX_X, MAX_Y = 1500, 1500

# max and min velocity
MIN_V, MAX_V = 0.1, 20.

# max waiting time
MAX_WT = 0.0

SIMULATION_TIME = 200

TRANSMISSION_RANGE = 200

CLUSTER_MOBILITY_DISTANCE = 100
CLUSTER_MAINTENANCE_N1 = 0.5
CLUSTER_MAINTENANCE_N2 = 0.5
CLUSTER_MAINTENANCE_RED_RED = 2

SIMULATION_MAP = {}

################################################################################
### VARIABLES FOR ANALYSIS OF ##################################################
# Mean Lifetime of CH
# Mean resident time
# Avg number of CH changes
# Avg number of cluster reaffiliations

# List to store the amount of time a node has spent as CH
A_CH_LIFETIME = [0]*N_NODES
A_TEMP_CH_TIME = [0]*N_NODES
A_NUM_OF_TIME_AS_CH = [0]*N_NODES

A_RESIDENT_TIME = [0]*N_NODES
A_TEMP_RESIDENT_TIME = [0]*N_NODES
A_NUM_OF_TIME_AS_RESIDENT = [0]*N_NODES

################################################################################
################################################################################

class Node():
    """
    Node colors : red, green and blue
    As per research paper specifications node colors were :

    red -> red
    yellow -> green
    white -> blue

    Colors were change for better visibility during simulation.
    """

    def __init__(self, nodeid):
        """
        Initialize values for different node variables
        """

        # Node identifier
        self.nodeid = nodeid
        # Records position of node at timestamp t
        self.history_cache = []
        # Current position (x,y) of node
        self.position = None
        # Current time
        self.curr_time = None

        # Color of node {red, blue, green}
        self.color = 'blue'
        # Membership table
        self.membership_table_CH = []
        self.membership_table_member = []

        # Neighbourhood set
        self.neighbour_set = {}

        # Check if nomination routine is run at least once
        self.if_nomination_done = False

    def discover_neighbours(self):
        """
        Find neighbours
        """

        t = time.time()

        # Remove neighbours from neighbourhood set which are not in
        # transmission range
        for nid in list(self.neighbour_set):
            d = np.linalg.norm(SIMULATION_MAP[nid].position-self.position)
            if d > TRANSMISSION_RANGE:
                del self.neighbour_set[nid]

        # Add new neighbours which have come in transmission range
        for nid in SIMULATION_MAP:
            d = np.linalg.norm(SIMULATION_MAP[nid].position-self.position)
            if d <= TRANSMISSION_RANGE:
                if nid not in self.neighbour_set and nid!=self.nodeid:
                    self.neighbour_set[nid] = t

    def check_mobility_info(self):
        """
        Check for drastic changes in mobility info
        """

        if len(self.history_cache) == 0:
            # print('k')
            self.history_cache.append(self.record_history())

        else:

            # Get last position and find (dx,dy)
            delta_p = self.position - self.history_cache[-1]['position']
            # Find sqrt(dx**2 + dy**2)
            D = np.linalg.norm(delta_p)

            # If there is significant change in position, update LDTSD
            # and run maintenance routine
            if D >= CLUSTER_MOBILITY_DISTANCE:

                t = time.time()
                prev_t = self.history_cache[-1]['time']
                S = D/(t-prev_t)
                THETA = self.get_theta(delta_p)
                self.curr_time = t

                self.S_i = S
                self.THETA_i = THETA

                self.history_cache.append(self.record_history(t))
                self.discover_neighbours()
                self.calculate_ldtsd()

                # Check if LDTSD is calculated
                if hasattr(self, 'LDTSD'):
                    self.run_maintenance_process()

    def get_theta(self, delta_p):
        """
        Get arc tan of dy/dx
        """

        if delta_p[0] > 0:
            r = np.absolute(delta_p[1]/delta_p[0])
            theta = np.arctan(r)
            return theta*np.sign(delta_p[1])

        elif delta_p[0] == 0:
            return np.pi*np.sign(delta_p[1])/2.0

        elif delta_p[0] < 0:
            r = np.absolute(delta_p[1]/delta_p[0])
            theta = np.arctan(r)
            return (np.pi-theta)*np.sign(delta_p[1])

    def record_history(self, t=None):
        """
        Create a history dictionary
        """

        history_record = {}
        if t!=None:
            history_record['time'] = t
        else:
            history_record['time'] = time.time()
        history_record['position'] = self.position

        return history_record

    def calculate_ldtsd(self):
        """
        Calculate the linear distance based total spatial dependency.
        Determines the stability of node.

        RD(j) = cos(theta_i - theta_j)
        SR(j) = min(S_i, S_j)/max(S_i, S_j)
        LDTSD(i) = transpose(RD)*SR or dot product
        """

        n_neighbour = len(self.neighbour_set)

        # If there are zero neighbours
        if n_neighbour == 0:
            self.LDTSD = -5.0

        else:
            RD = np.zeros(n_neighbour)
            SR = np.zeros(n_neighbour)

            i = 0
            for nid in self.neighbour_set:

                if hasattr(SIMULATION_MAP[nid], 'THETA_i'):
                    RD[i] = np.cos(self.THETA_i - SIMULATION_MAP[nid].THETA_i)
                else:
                    # If THETA_i of neighbour is not calculated yet due to no
                    # significant change in it's position
                    RD[i] = np.cos(self.THETA_i)

                if hasattr(SIMULATION_MAP[nid], 'S_i'):
                    if max(self.S_i, SIMULATION_MAP[nid].S_i) != 0:
                        SR[i] = min(self.S_i, SIMULATION_MAP[nid].S_i)/max(self.S_i, SIMULATION_MAP[nid].S_i)
                else:
                    avg_v = (MIN_V+MAX_V)/2.0
                    SR[i] = min(self.S_i, avg_v)/max(self.S_i, avg_v)

                i += 1

            self.LDTSD = np.dot(RD, SR)

    def run_nomination_process(self):
        """
        If node has the largest LDTSD among all its white neighbors, it turns to
        red to form a new cluster labeled with the ID number of this node as the
        cluster ID.
        """

        nominate = False

        if len(self.neighbour_set) == 0:
            nominate = True

        else:

            for nid in self.neighbour_set:
                if not hasattr(SIMULATION_MAP[nid], 'LDTSD'):
                    return

            for nid in self.neighbour_set:
                if self.LDTSD < SIMULATION_MAP[nid].LDTSD:
                    nominate = False
                    break
                else:
                    nominate = True

        # print(self.nodeid, len(self.neighbour_set), self.LDTSD, nominate)

        if nominate:
            t = time.time()
            self.color = 'red'
            self.membership_table_member = []
            for nid in self.neighbour_set:
                self.membership_table_member.append(nid)
                SIMULATION_MAP[nid].color = 'green'
                SIMULATION_MAP[nid].membership_table_CH.append(self.nodeid)


    def run_maintenance_process(self):
        """
        Run maintenance process
        """

        if self.color == 'red':
            self.maintain_red()

        if self.color == 'green':
            self.maintain_green()

        if self.color == 'blue':
            self.maintain_blue()

    def maintain_red(self):
        """
        For reference, check maintenance routine for node when it's color is red
        in research paper.
        """

        self.update_membership_table_member()

        n_blue = 0.0
        n_nodes = 0.0

        # Checks for existence of any red-red direct connnection
        check_for_red_red = False

        for nid in self.neighbour_set:
            if SIMULATION_MAP[nid].color == 'red':
                if (self.curr_time-self.neighbour_set[nid]) > CLUSTER_MAINTENANCE_RED_RED:
                    check_for_red_red = True
                    if self.LDTSD < SIMULATION_MAP[nid].LDTSD:
                        self.color = 'green'
                        self.membership_table_member = []
                        self.membership_table_CH.append(nid)
                        SIMULATION_MAP[nid].membership_table_member.append(self.nodeid)
                    else:
                        SIMULATION_MAP[nid].color = 'green'
                        SIMULATION_MAP[nid].membership_table_member = []
                        SIMULATION_MAP[nid].membership_table_CH.append(self.nodeid)
                        self.membership_table_member.append(nid)

            elif SIMULATION_MAP[nid].color == 'blue':
                n_blue += 1.0

            n_nodes += 1.0

        if not check_for_red_red:
            r = 0.0
            # if number of nodes in neighbourhood is zero, white nodes ratio or
            # r will be undefined
            if n_nodes == 0:
                # Setting it to less than n1 so that node remains red when
                # there are no nodes in neighbourhood
                r = CLUSTER_MAINTENANCE_N1 - 0.1
            else:
                r = n_blue/n_nodes
            if r >= CLUSTER_MAINTENANCE_N1:
                self.color = 'blue'
                self.membership_table_member = []

    def maintain_green(self):
        """
        For reference, check maintenance routine for node when it's color is
        yellow in research paper.
        """

        # Flag for checking if a red node exists in neighbourhood
        if_red_not_exists = True

        # Flag for checking if all red nodes in neighbourhood have lower LDTSD
        if_all_red_has_lower_ldtsd = True

        # List for maintaining ids of red nodes having lower LDTSD than
        # current node
        nlist = []

        for nid in self.neighbour_set:
            if SIMULATION_MAP[nid].color == 'red':
                if_red_not_exists = False
                if if_all_red_has_lower_ldtsd:
                    if self.LDTSD > SIMULATION_MAP[nid].LDTSD:
                        nlist.append(nid)
                    else:
                        if_all_red_has_lower_ldtsd = False

        if if_red_not_exists:
            for nid in self.membership_table_CH:
                if self.nodeid in SIMULATION_MAP[nid].membership_table_member:
                    SIMULATION_MAP[nid].membership_table_member.remove(self.nodeid)
            self.membership_table_CH = []
            self.color = 'blue'

        elif if_all_red_has_lower_ldtsd:
            for nid in nlist:
                if nid in self.membership_table_CH:
                    self.membership_table_CH.remove(nid)
            if len(self.membership_table_CH) == 0:
                self.color = 'blue'


    def maintain_blue(self):
        """
        For reference, check maintenance routine for node when it's color is
        white in research paper.
        """

        # Flag for checking if there exists a red node with larger LDTSD in
        # imediate neigbourhood
        if_larger_red_exists = False

        n_blue = 0.0
        n_nodes = 0.0

        for nid in self.neighbour_set:
            if not if_larger_red_exists:
                if SIMULATION_MAP[nid].color == 'red':
                    if self.LDTSD < SIMULATION_MAP[nid].LDTSD:
                        if_larger_red_exists = True
                        self.color = 'green'
                        self.membership_table_CH.append(nid)
                        SIMULATION_MAP[nid].membership_table_member.append(self.nodeid)

            if SIMULATION_MAP[nid].color == 'blue':
                n_blue += 1.0

            n_nodes += 1.0

        if not if_larger_red_exists:
            r = 0
            # if number of nodes in neighbourhood is zero, white nodes ratio or
            # r will be undefined
            if n_nodes == 0:
                # Setting it to more than n2 so that node does not remain white
                # there are no nodes in neighbourhood. Node will try to initiate
                # nomination process when no nodes are found in neighbourhood
                r = CLUSTER_MAINTENANCE_N2 + 0.1
            else:
                r = n_blue/n_nodes
            if r >= CLUSTER_MAINTENANCE_N2:
                self.run_nomination_process()

    def update_membership_table_member(self):
        """
        Keep only those members in membership table which are still neighbours
        and are colored green (yellow)
        """

        temp = []

        for nid in self.membership_table_member:
            if nid in self.neighbour_set:
                if SIMULATION_MAP[nid].color == 'green':
                    temp.append(nid)

        self.membership_table_member = temp

def performance_analysis_per_step(curr_t):
    """
    Function to measure performance of algorithm in terms of different
    parameters
    """

    for i in range(N_NODES):

        if A_TEMP_CH_TIME[i] == 0:
            if SIMULATION_MAP[i].color == 'red':
                A_TEMP_CH_TIME[i] = curr_t
                A_NUM_OF_TIME_AS_CH[i] += 1
        else:
            if SIMULATION_MAP[i].color != 'red':
                A_TEMP_CH_TIME[i] = curr_t - A_TEMP_CH_TIME[i]
                A_CH_LIFETIME[i] += A_TEMP_CH_TIME[i]
                A_TEMP_CH_TIME[i] = 0

        if A_TEMP_RESIDENT_TIME[i] == 0:
            if SIMULATION_MAP[i].color == 'green':
                A_TEMP_RESIDENT_TIME[i] = curr_t
                A_NUM_OF_TIME_AS_RESIDENT[i] += 1
        else:
            if SIMULATION_MAP[i].color != 'green':
                A_TEMP_RESIDENT_TIME[i] = curr_t - A_TEMP_RESIDENT_TIME[i]
                A_RESIDENT_TIME[i] += A_TEMP_RESIDENT_TIME[i]
                A_TEMP_RESIDENT_TIME[i] = 0


def aggregate_performance_param(curr_t):
    """
    Function to aggregate performance params collected over all steps till time
    curr_t

    mean_lifetime_ch : cluster lifetime, is calculated by cumulating the time
    duration of all nodes being CHs and dividing it by the total number of CHs
    during one simulation run.

    mean_resident_time : is the average time interval that a node affiliates
    with a certain cluster.
    """

    mean_lifetime_ch = 0
    try:
        mean_lifetime_ch = sum(A_CH_LIFETIME)/sum(A_NUM_OF_TIME_AS_CH)
    except ZeroDivisionError as e:
        print(e)
    print("Mean lifetime CH till {} : ".format(curr_t), mean_lifetime_ch)

    mean_resident_time = 0
    try:
        mean_resident_time = sum(A_RESIDENT_TIME)/sum(A_NUM_OF_TIME_AS_RESIDENT)
    except ZeroDivisionError as e:
        print(e)
    print("Mean resident time till {} : ".format(curr_t), mean_resident_time)


if __name__ == "__main__":

    from mobility_models import reference_point_group
    import sys

    # Fix the random seed
    np.random.seed(0xffff)
    model = None

    # Create node objects with their corresponding node id
    for i in range(N_NODES):
        SIMULATION_MAP[i] = Node(i)

    if sys.argv[1] == 'rpg':
        groups = [4 for _ in range(10)]
        rpg = reference_point_group(
            groups, dimensions=(MAX_X, MAX_Y), aggregation=0.2)
        model = rpg

    DRAW = True
    CLUSTER_LINE = True

    if DRAW:
        import matplotlib.pyplot as plt
        plt.ion()
        ax = plt.subplot(111)
        ax.set_xlim(0,MAX_X)
        ax.set_ylim(0,MAX_Y)
        sc = ax.scatter([], [], linestyle='-', marker='.')

        if CLUSTER_LINE:
            for l in range(N_NODES):
                ax.plot([], [], 'b-')

    INIT_TIME = time.time()
    steps = 0
    prev_time = INIT_TIME

    PERFORMANCE_SAMPLE_T = [20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0]
    IF_SAMPLE_RECORDED = [False]*10
    SAMPLE_COUNTER = 0

    for xy in model:

        for i in range(N_NODES):
            SIMULATION_MAP[i].position = xy[i]

        curr_time = time.time()
        t0 = curr_time - prev_time

        # Do some calc for performance analysis
        performance_analysis_per_step(curr_time)

        # Sampling interval
        if t0 > 0.001:

            for i in range(N_NODES):
                SIMULATION_MAP[i].check_mobility_info()

            prev_time = curr_time

            if DRAW:
                if CLUSTER_LINE:
                    lno = 0
                    for i in range(N_NODES):
                        if SIMULATION_MAP[i].color == 'red':
                            # Draw lines between CH and it's members
                            for nid in SIMULATION_MAP[i].membership_table_member:
                                ax.lines[lno].set_data([xy[i,0],xy[nid,0]], [xy[i,1],xy[nid, 1]])
                                lno += 1
                    for i in range(lno, N_NODES):
                        ax.lines[i].set_data([],[])

                color = []
                # Scatter plot for nodes with different colors based on thier
                # role
                for i in range(N_NODES):
                    color.append(SIMULATION_MAP[i].color)
                sc.set_offsets(np.column_stack((xy[:,0],xy[:,1])))
                sc.set_color(color)
                plt.draw()
                plt.pause(1e-17)

        time.sleep(0.00001)

        #if steps%200 == 0:
        #    print(steps, round(curr_time-INIT_TIME,2))

        steps += 1

        if SAMPLE_COUNTER < 10:
            if not IF_SAMPLE_RECORDED[SAMPLE_COUNTER]:
                if round(curr_time-INIT_TIME, 1) == PERFORMANCE_SAMPLE_T[SAMPLE_COUNTER]:
                    aggregate_performance_param(curr_time-INIT_TIME)
                    IF_SAMPLE_RECORDED[SAMPLE_COUNTER] = True
                    SAMPLE_COUNTER += 1

        if curr_time - INIT_TIME > SIMULATION_TIME:
            break
