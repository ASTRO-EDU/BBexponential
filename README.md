# BBexponential
In this repo there is the implementation for Bayesian Blocks with exponential fitting function. It has been implemented by adding the class ExponentialBlocks_Events in bb_exponential.

This is a brief explaination of the files in the repo:
* bb_exponential: the main file where the most central functions are located, here there is the class ExponentialBlocks_Events and the function test_bb_exp; the first is used for the bayesian_blocks function in Astropy, the second to test an exponential fitting function by checking the graph, the used time for calculation and the block extremes
* reader_class: a small class used to obtain binned data from h5 files
* reader: python notebok used for development, also useful to see how the Reader class is used in practice
* bayesian_blocks_exp: python notebook used for the development, also useful to see how ExponentialBlocks_Events is used in practice
* ncp_prior: deprecated, python script used for development, used to understand the ideal structure of ncp_prior for artificial data
* ncp_prior_optimize: deprecated, python notebook used for development, used to understand the ideal structure of ncp_prior for artificial data
* ncp_2: python notebook used for development, it has been used to understand the rough structure of the iperparameter ncp_prior in real data
* ncp_prior_v2: python script used for development, it has been used to understand the rough structure of the iperparameter ncp_prior in real data
* reader_test: python script used for development, it has been used to understand the precise structure of ncp_prior in real data. Very useful if there is the need to modify ncp_prior for different type of data or if different specifications for bad behaviour are constructed
