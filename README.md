# Intelligent Perception and Decision Making: Coursework Summary

This repository contains homework assignments (HW1 to HW4) for a course on Intelligent Perception and Decision Making. Each homework focuses on different aspects of computer vision, machine learning, robotics, and path planning.

## HW1: Bird's Eye View (BEV) Projection and ICP Alignment

### Task 1: BEV Projection
- Utilize `load.py` to collect data and `bev.py` for BEV projection.
- Capture screenshots and project points from BEV image to the front image.

### Task 2: ICP Alignment
- Collect data using `load.py` and reconstruct point clouds with `reconstruct.py`.
- Employ ICP (Iterative Closest Point) algorithm for environmental reconstruction and trajectory creation of ground truth and estimated camera poses.

[HW1 Details](https://github.com/henry890112/Intelligent-perception-and-decision-making/blob/main/Tuesday_hw1/README.md)

## HW2: Data Generation, Transformation, and Model Training

- Generate data divided into test, train, and annotations.
- Transform data into `.odgt` format and train custom models.
- Collect RGB, depth, and annotation data in the Habitat environment.
- Produce predicted semantic segmentation and colorize ground truth and estimated annotations.
- Reconstruct the environment using semantic segmentation.

[HW2 Details](https://github.com/henry890112/Intelligent-perception-and-decision-making/blob/main/Tuesday_hw2/README.md)

## HW3: BEV Imaging and Path Planning with RRT

- Obtain BEV images of the entire environment.
- Plan paths using the RRT (Rapidly-exploring Random Tree) algorithm.
- Display paths in the Habitat environment and save the path planning results.

[HW3 Details](https://github.com/henry890112/Intelligent-perception-and-decision-making/blob/main/Tuesday_hw3/README.md)

## HW4: Robot Manipulation Framework

### Task 1: Forward Kinematic Algorithm and Jacobian Matrix
- Implement and test forward kinematic algorithms and Jacobian matrices for each pose.

### Task 2: Inverse Kinematic Algorithm
- Implement and test the inverse kinematic algorithm.
- Complete manipulation pipeline including pose matching, robot movement control using inverse kinematics, and motion planning with RRT-Connect for collision-free paths.

[HW4 Details](https://github.com/henry890112/Intelligent-perception-and-decision-making/blob/main/Tuesday_hw4/README.md)

---

This README provides an overview of the assignments and their objectives in the field of intelligent perception and decision making. Each link directs to the detailed README of the respective homework for further information.
