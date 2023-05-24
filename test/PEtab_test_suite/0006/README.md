# PEtab test case 0006

## Objective

This case tests support for time-point specific overrides in the measurement
table.

The model is to be simulated for a single experimental condition. The single
model output is scaled by a different parameter at each timepoint.

## Model

A simple conversion reaction `A <=> B` in a single compartment, following
mass action kinetics.
