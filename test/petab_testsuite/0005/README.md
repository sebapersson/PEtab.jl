# PEtab test case 0005

## Objective

This case tests support for parametric overrides from condition table.

The model is to be simulated for two different experimental conditions
(here: different initial concentrations). The observable is offsetted by
a parametric override in the condition table (i.e. the actual value has
to be taken from the parameter table).

## Model

A simple conversion reaction `A <=> B` in a single compartment, following
mass action kinetics.
