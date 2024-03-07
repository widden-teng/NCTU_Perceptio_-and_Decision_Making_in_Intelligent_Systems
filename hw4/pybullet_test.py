import numpy as np
import pybullet as p
import time
import pybullet_data
import _thread
import threading


DURATION = 10000
ALPHA = 1000

def new_obj(deley, velocity):
    time.sleep(deley)
    DURATION = 10000
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    p.setGravity(0, 0, -10)
    cubeStartPos = [0, 0, 1]
    cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    boxId = p.loadURDF("r2d2.urdf", cubeStartPos, cubeStartOrientation)
    for i in range(DURATION):
        p.stepSimulation()
        time.sleep(1./240.)
        boxPos, boxOrn = p.getBasePositionAndOrientation(boxId)
        p.resetBaseVelocity(boxId,linearVelocity=[velocity,0,0])
    p.disconnect()


physicsClient = p.connect(p.GUI)
# for i in range (2, 30, 2):
#     print(i)
#     _thread.start_new_thread( new_obj, (i,2,) )
_thread.start_new_thread( new_obj, (0,2,) )
_thread.start_new_thread( new_obj, (2,2,) )
_thread.start_new_thread( new_obj, (4,2,) )
while 1:
   pass
