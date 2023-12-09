# Snow_simulation_experiment
## Demo
  ![image](https://github.com/shigon255/Snow_simulation_experiment/blob/main/snow_simulation_reference.gif?raw=true)
## Introduction
  + This is the final project of NYCU Computer Animation and Special Effects, 2023 Spring.
  + In this project, we will investigate the usage of the Material Point Method (MPM) for simulating snow. MPM is a method for simulating continuous materials, and we will study its techniques and principles.
  + Next, we will delve into MLS-MPM (Moving Least Square MPM), which is an enhancement of the MPM method designed to improve its stability and simulation efficiency. We will explore how MLS-MPM enhances MPM.
  + Finally, we will implement snow simulation using both MPM and MLS-MPM. We will use the Taichi programming language for simulation, utilizing the provided mpm3d_ggui.py as the foundation for generating particle animations. These animations will then be input into Houdini animation software to create the snow's visual representation. We will showcase our simulation results and discuss the influence of different parameters on the simulation outcomes.
  + In addition to simulating snow using MPM and MLS-MPM, we will also investigate how to handle the behavior of phase-change materials when they melt and solidify. We have implemented a method called Augmented MPM, which can handle temperature and phase-change to achieve the desired simulation effects. However, the numerical system is currently unstable. We will explain the principles and techniques behind Augmented MPM.
    + For phase-change material, please refer to [this repo](https://github.com/shigon255/Phase_change_material)
## Usage
  + Install taichi package
## Experiment result
  + Please refer to report: snow_simulation.pdf
  + Result videos are in [this youtube playlist](https://www.youtube.com/watch?v=Cxg7x7qMWxk&list=PLTNy_HFJIhinfaYEZrb2ORpM6dxey9AOk)
