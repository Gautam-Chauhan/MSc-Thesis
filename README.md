#### Introduction

This is a repo containing the code used in my MSc thesis: Monte Carlo methods for counting linear regions in neural networks!

#### Files

This repo contains the following files:
- `iid_uniform_code.ipynb`: This Jupyter notebook contains experiments related to using a binomial point process (sampling a fixed number of points independently and uniformly) to count linear regions in the input space of the neural network
- `gbn_bounded_current.py`: This code contains functions to generate Gaussian Blue Noise with a fixed number of points (adapted into python from C++ code found here: https://abdallagafar.com/publications/gbn/)
- `spectrum_nd.py`: This code calculates the power spectrum/periodogram of a set of Gaussian Blue Noise points (adapated into python from C++ code found here: https://abdallagafar.com/publications/gbn/)
- `GBN_sampling.ipynb`: This Jupyter notebook uses the functions provided in `gbn_bounded_current.py` and `spectrum_nd.py` to generate and test the properties of a sample of Gaussian Blue Noise
