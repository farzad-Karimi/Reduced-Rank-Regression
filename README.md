# Reduced-Rank-Regression

This repository contains the Reduced Rank Regression (RRR) technique applied to Neuropixels change detection task recordings from the Allen Brain Institute.

The code was run on the ARC cluster, and the corresponding Slurm file is available.

The animal IDs for familiar and novel images are saved separately in different files.

reduced_rank_regressor.py: Includes the RRR method functions.

funcs.py: Contains the code for extracting spiking data using the Allen API.

RRR_lick.py: Includes the main functions.

In this version of the code, the RRR has been applied to pre- and post-lick trials of 250 ms duration.
