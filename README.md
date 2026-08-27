#### Introduction

Welcome! This repository contains the code used for my MSc thesis: Monte Carlo methods for counting linear regions in neural networks.

#### Files

- `brute_force_gaussian_blue_noise.ipynb`: This notebook was used to generate blue noise points for my experiments, based on the paper Gaussian Blue Noise: https://abdallagafar.com/publications/gbn/. I would like to credit Armand de Caqueray for providing the baseline code for this. It contains a series of experiments to find the best learning rate for each dimension, and code to generate 100 independent blue noise point sets over a range of sample sizes and dimensions.

- `scenario_1.ipynb`: This notebook contains a series of experiments to test the performance of the different sampling methods (uniform, Halton, Sobol, Blue Noise and grid) on a series of neural networks where the linear regions form hypercubes (see experiment 1 in the thesis).

- `scenario_2.ipynb`: This notebook contains a series of experiments to test the performance of the different sampling methods on a series of neural networks where the linear regions form elongated hyperrectangles (see experiment 2 in the thesis).

- `scenario_3.ipynb`: This notebook contains a series of experiments to test the performance of the different sampling methods on a series of neural networks where the linear regions form differently sized hypercubes and hyperrectangles (see experiment 3 in the thesis).

#### How to run 

- To run `brute_force_gaussian_blue_noise.ipynb`, open the file in Google Colaboratory and select a GPU runtime. Due to computational cost, a CPU is not suitable for running this code, and use of a powerful GPU is recommended (I used A100 GPUs).
- To run `scenario_1.ipynb`, `scenario_2.ipynb` and `scenario_3.ipynb`:
1. Obtain a set of blue noise points, either by running the experiments in `brute_force_gaussian_blue_noise.ipynb` or by downloading them from my google drive: https://drive.google.com/drive/folders/16e7BFBGsPTdXqzWZiyOvVpRaYD4OLaEY?usp=drive_link.
2. If running locally, create a new virtual environment and install the dependencies listed in requirements.txt.
3. Get the file path to the location where the blue noise point sets are stored on your machine, and replace the path in each blue noise related cell with your path.
4. Run the cells sequentially. Note that the cells in each block or experiment must be run in sequence to exactly reproduce the results shown, as a pseudo-random number generator is initialised at the start of each experiment. CPU can be used to run these experiments.
