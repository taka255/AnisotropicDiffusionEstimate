# AnisotropicDiffusionEstimate

[![Build Status](https://github.com/taka255/AnisotropicDiffusionEstimate.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/taka255/AnisotropicDiffusionEstimate.jl/actions/workflows/CI.yml?query=branch%3Amain)


This Julia package is designed for the estimation of two translational diffusion coefficients and one rotational diffusion coefficient from the trajectories of the center of mass of an ellipsoid undergoing anisotropic diffusion, incorporating error estimation. It employs a particle filter implementation of the belief propagation method, based on maximum likelihood estimation. For further details, please refer to this [paper](https://arxiv.org/abs/2401.15909). A simple executable example can be found at [here](https://github.com/taka255/AnisotropicDiffusionEstimate/tree/main/examples).