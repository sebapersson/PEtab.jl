# PEtab test case 0004

## Objective

This case tests support for parametric observable parameter overrides in
measurement tables

Simulated data describes measurements with different offset and scaling
parameters for a single observable. These values of the respective
(non-estimated) parameters referenced in `observableParameters` need to be
looked up in the parameter table to replace the placeholders in
`observableFormula`.

## Model

A simple conversion reaction `A <=> B` in a single compartment, following
mass action kinetics.
