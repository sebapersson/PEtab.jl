# Test Case 039

Test case for when a feed-forward neural network appears in one of the observable formulas,
and one of the ML model input arguments is given via an array file where the array file
contains condition specific values.

## Model Structure

The SBML model for this problem is given as:

$$\frac{\mathrm{d} \text{prey}}{\mathrm{d} t} = \alpha \cdot \text{prey} - \beta \cdot \text{prey} \cdot \text{predator} $$

$$\frac{\mathrm{d} \text{predator}}{\mathrm{d} t} = \gamma \cdot \text{prey} \cdot \text{predator} - \delta \cdot \text{predator}$$

And the observable formula for `prey` is given by `NN[1] / 0.9 + prey` (the output from a
neural network).

## Data-Driven Model Structure

`N` is a feed-forward neural network with the following specification:

| LayerID | Layer                                                   | Activation Function |
|---------|---------------------------------------------------------|---------------------|
| layer1  | Linear(in_features = 4, out_features = 5, bias = true)  | relu                |
| layer2  | Linear(in_features = 5, out_features = 10, bias = true) | relu                |
| layer3  | Linear(in_features = 10, out_features = 1, bias = true) | identity            |

The inputs to the network are given by `prey` (argument 0) and an array file (argument 1),
where the array file contains condition specific values.
