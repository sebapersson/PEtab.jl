# PEtab test case 0010

## Objective

This case tests support for partial preequilibration.

The model is to be simulated for a preequilibration condition and a
simulation condition.
Species `B` is to be reinitialized after preequilibration.
For `A` the preequilibration result is to be used.

## Model

A simple conversion reaction `A <=> B` in a single compartment, following
mass action kinetics.
