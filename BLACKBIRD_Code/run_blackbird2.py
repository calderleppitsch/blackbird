"""
mavsimPy
    - Chapter 3 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        12/18/2018 - RWB
        1/14/2019 - RWB
"""
import sys
sys.path.append('/Users/C/Dropbox/work/blackbird/UAVBook_references')
sys.path.append('/Users/C/Dropbox/work/blackbird')
import numpy as np
import matplotlib.pyplot as plt

import BLACKBIRD_Code.blackbird_params2 as PMT
from BLACKBIRD_Code.blackbird_viewer import MavViewer
from BLACKBIRD_Code.blackbird_dynamics_v2 import MavDynamics
from message_types.msg_delta import MsgDelta

###########################################################
#Time Variables
ts_simulation = 0.01  # smallest time step for simulation
start_time = 0.  # start time for simulation
end_time = 400.  # end time for simulation

ts_video = 0.1  # write rate for video
ts_control = ts_simulation  # sample rate for the controller
###########################################################

#Initialize Architecture
blackbird = MavDynamics(ts_simulation)
delta = MsgDelta()

#Initialize State Arrays
x = [PMT.north0]
z = [PMT.down0 * -1]
y = [PMT.east0]
phi = [PMT.phi0]
theta = [PMT.theta0]
psi = [PMT.psi0]
e0 = [PMT.e0]
e1 = [PMT.e1]
e2 = [PMT.e2]
e3 = [PMT.e3]
p = [PMT.p0]
q = [PMT.q0]
r = [PMT.r0]
Va = [PMT.Va0]
alpha = [0]
beta = [0]


# initialize the visualization
VIDEO = False  # True==write video, False==don't write video
mav_view = MavViewer()  # initialize the mav viewer
if VIDEO is True:
    from chap2.video_writer import VideoWriter
    video = VideoWriter(video_name="chap3_video.avi",
                        bounding_box=(0, 0, 1000, 1000),
                        output_rate=ts_video)

#Run the simulation
wind=np.zeros((6,1))

sim_time = start_time
while sim_time < end_time:
    delta.from_array(np.array([-0.4, 0.0, 0.0, 0.5]))
    blackbird.update(delta.to_array(),wind)
    mav_view.update(blackbird.true_state)
    if VIDEO is True:
        video.update(sim_time)
    sim_time += ts_simulation

if VIDEO is True:
    video.close()




