# PEtab test case 0018

## Objective

This case tests support for RateRules and partial preequilibration with `NaN`'s
in the condition file.

The model is to be simulated for a preequilibration condition and a
simulation condition.
For preequilibration, species `B` is initialized with `0`. For simulation,
`B` is set to `NaN`, meaning that it is initialized with the result from
preequilibration.
`A` is reinitialized to the value in the condition table after
preequilibration.

## Model

A simple conversion reaction `A <=> B` in a single compartment, following
mass action kinetics. Dynamics of are specified as `RateRule`s targeting a
parameter and a species.
