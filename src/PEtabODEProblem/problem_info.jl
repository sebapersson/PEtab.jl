const GRADIENT_METHODS = [nothing, :ForwardDiff, :ForwardEquations, :Adjoint]
const HESSIAN_METHODS = [nothing, :ForwardDiff, :BlockForwardDiff, :GaussNewton]
const FIM_METHODS = [nothing, :ForwardDiff, :GaussNewton]
