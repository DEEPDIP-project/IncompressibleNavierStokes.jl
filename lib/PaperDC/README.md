# PaperDC

Scripts for generating results of the paper
[Discretize first, filter next: learning divergence-consistent closure models for large-eddy simulation](https://arxiv.org/abs/2403.18088).

## Set up environment

From this directory, run:

```sh
julia setup.jl
```

Now you can run the scripts in this directory:

- `prioranalysis.jl`: Generate results for section 5.1 "Filtered DNS (2D and 3D)"
- `postanalysis.jl`: Generate results for section 5.2 "LES (2D)"
