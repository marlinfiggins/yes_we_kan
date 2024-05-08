# Yes, we KAN

This is a repository implementing several extensions of KAN, a neural network architecture developed in [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756) by Liu et al.

Note: What I call a KAN layer does not follow the exact implementation given in the KAN paper.
To avoid tracking and updating a grid, I instead transform outputs into [0, 1] and define the B-Spline over [0, 1].