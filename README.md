# ARC Stability Landscape — Computational Model

This repository contains a set of Jupyter notebooks that implement and analyze a stochastic, resource-explicit model of bacterial competition between a reference strain and a synthetic library of ARC-bearing competitors.  
The notebooks reproduce the complete modeling workflow, from parameter estimation to simulation and analysis, as described in the Supplementary Information.

## Notebook Structure

### 1. `MonodGillespieARC_parametrizationStrain.ipynb`
Fits the growth parameters of individual strains from experimental growth curves under aerobic and anaerobic conditions.  
The resulting parameters ($V_i$, $K_i$, $c_i$) are stored and used as inputs for the simulations.

### 2. `MonodGillespieMI_stability.ipynb`
Describes and illustrates the computational model.  
This notebook presents the stochastic Monod–Gillespie framework, the implementation of growth–dilution cycles, and validation runs using reference simulations.

### 3. `MonodGillespieARC_stability_expe.ipynb`
Executes the in silico competition experiments that reproduce the ARC–control assays.  
Simulations include 1000 ARC-bearing strains across environmental regimes: constant aerobiosis, constant anaerobiosis, or a single switch from anaerobiosis to aerobiosis after *K* days.  
The outputs include daily abundances, final frequencies, and persistence statistics.

### 4. `MonodGillespieARC_stability_analysis.ipynb`
Loads the simulation results, aggregates replicates, and computes population-level metrics such as persistence and rescue probabilities.  
This notebook also generates the figures and summary statistics used in the main text and Supplementary Information.
