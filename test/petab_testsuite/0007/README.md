# PEtab test case 0007

## Objective

This case tests support for observable transformations to log10 scale.

The model is to be simulated for a single experimental condition. PEtabMeasurements
for observable `obs_a` are to be used as is, measurements for `obs_b` are to
be transformed to log10 scale for computing chi2 and likelihood.

## Model

A simple conversion reaction `A <=> B` in a single compartment, following
mass action kinetics.
