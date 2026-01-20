# PEtab test case 0002

## Objective

This case tests support for multiple simulation conditions

The model is to be simulated for two different experimental conditions
(here: different initial concentrations).

For `b0`, `nan` is used in the condition table, indicating that the default
model values for `b0` should be used for either condition.

## Model

A simple conversion reaction `A <=> B` in a single compartment, following
mass action kinetics.
