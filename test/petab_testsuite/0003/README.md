# PEtab test case 0003

## Objective

This case tests support for numeric observable parameter overrides in
measurement tables

Simulated data describes measurements with different offset and scaling
parameters for a single observable. These respective numeric
`observableParameters`
from the measurement table have to be applied to the placeholders in
observableFormula.

## Model

A simple conversion reaction `A <=> B` in a single compartment, following
mass action kinetics.
