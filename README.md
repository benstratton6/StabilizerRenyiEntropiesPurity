# StabilizerRenyiEntropiesPurity
This contains code that supports the manuscript "An Algorithm for Estimating alpha-Stabilizer Rényi Entropies via Purity" - https://doi.org/10.48550/arXiv.2507.02540

In main, one can find functions to simulate the quantum circuits detailed in the manuscript. There is one function specifically for alpha=2 and another for all interger alpha that measures the alpha-SRE for the example state parameterised by theta. There is also a function to simulate the algorithm applied to the input states in https://doi.org/10.1038/s41534-022-00666-5. 

Data included in the plot in the paper is stored in the data directory. Note, there is an element of randomness in the simulation, so re-running the code will not return exactly the same data. 
