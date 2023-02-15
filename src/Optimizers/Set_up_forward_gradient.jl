using ProgressMeter

# TODO : Add ADAM to its own file 

mutable struct Adam{
    T1 <: AbstractArray{Float64},
    T2 <: Function,
    T3 <: Float64,
    T4 <: Vector{Float64},
    T5 <: Int64, 
    T6 <: Function}

    theta::T1                # Parameter array
    thetaOld::T1             # Old parameter array
    lB::T1                   # Lower bounds for theta
    uB::T1                   # Upper bounds for theta
    evalGradF::T2            # Gradient function
    cost::T3                 # value of the cost function
    m::T1                    # First moment
    v::T1                    # Second moment
    b1::T3                   # Exp. decay first moment
    b2::T3                   # Exp. decay second moment
    a::T4                    # list of (decaying) step sizes. Size of the number of iterations 
    eps::T3                  # Epsilon for stability 
    it::T5                   # Time step (iteration)
    β::T3                    # Reduction factor (scales the step length)
    r::T3                    # Reduction multiplier (shortens the step length at failed step)
    c::T3                    # Increas multiplyer (Increases the step length at successful step, up to 1.0)
    fail::T5                 # Number of infeasible points tested in a row
    evalF::T6                # Cost function 
end


"""
    createFowardGradientProb(petabProblem::PEtabODEProblem,
                             stepLength, 
                             nIt; 
                             b1=0.99, b2=0.99)::Adam

    For a PeTab-model in its optimization struct petabProblem create an ADAM struct 
    where we have full batch optimisation where the gradient is computed via 
    forward gradient. For ADAM optmizer the stepLength (e.g 1e-3 or a vector with 
    the length of nIt) and nIt are further specified here. 

    The API with ADAM will be changed to not restrict the usage of ADAM to forward 
    gradient.
"""
function createFowardGradientProb(petabProblem::PEtabODEProblem,
                                  stepLength, 
                                  nIt; 
                                  b1=0.99, b2=0.99)::Adam
    
    # In case step-length is provided as scalar assume constant step lenght, else 
    # use the user provided stepLength list. 
    a::Array{Float64, 1} = Array{Float64, 1}(undef, nIt)
    if typeof(stepLength) <: Number
        a .= [stepLength for i in 1:nIt]
    elseif length(stepLength) == nIt
        a .= stepLength
    else
        @printf("Error : If step-length is an array its lenght must match the number of iterations, 
                 currently length(stepLength) %d and nIt = %d\n", length(stepLength), nIt)
    end

    # Pre-allocate arrays to hold model parameters 
    nParamEst = petabProblem.nParametersToEstimate
    thetaOld::Array{Float64, 1} = Array{Float64, 1}(undef, nParamEst)
    theta::Array{Float64, 1} = Array{Float64, 1}(undef, nParamEst)

    # Function calculating unbiased gradient estimate 
    evalGradF = (pVec) -> calcUnbiasedGrad(pVec, petabProblem.computeCost)
    
    # ADAM optimizer parameters and arrays  
    loss = 0.0
    b1 = b1
    b2 = b2
    m = zeros(nParamEst)
    v = zeros(nParamEst)
    eps = 1e-8
    it = 0
    β = 1.0
    r = 0.2
    c = 1.3
    fail = 0
    lB = petabProblem.lowerBounds
    uB = petabProblem.upperBounds

    return Adam(theta, thetaOld, lB, uB, evalGradF, loss, m, v, b1, b2, a, eps, it, β, r, c, fail, petabProblem.computeCost)
end
function createFowardGradientProb(evalF::Function,
                                  stepLength, 
                                  nParamEst,
                                  nIt, 
                                  lB, 
                                  uB; 
                                  b1=0.99, b2=0.99)::Adam
    
    # In case step-length is provided as scalar assume constant step lenght, else 
    # use the user provided stepLength list. 
    a::Array{Float64, 1} = Array{Float64, 1}(undef, nIt)
    if typeof(stepLength) <: Number
        a .= [stepLength for i in 1:nIt]
    elseif length(stepLength) == nIt
        a .= stepLength
    else
        @printf("Error : If step-length is an array its lenght must match the number of iterations, 
                 currently length(stepLength) %d and nIt = %d\n", length(stepLength), nIt)
    end

    # Pre-allocate arrays to hold model parameters 
    thetaOld::Array{Float64, 1} = Array{Float64, 1}(undef, nParamEst)
    theta::Array{Float64, 1} = Array{Float64, 1}(undef, nParamEst)

    # Function calculating unbiased gradient estimate 
    evalGradF = (pVec) -> calcUnbiasedGrad(pVec, evalF)
    
    # ADAM optimizer parameters and arrays  
    loss = 0.0
    b1 = b1
    b2 = b2
    m = zeros(nParamEst)
    v = zeros(nParamEst)
    eps = 1e-8
    it = 0
    β = 1.0
    r = 0.2
    c = 1.3
    fail = 0

    return Adam(theta, thetaOld, lB, uB, evalGradF, loss, m, v, b1, b2, a, eps, it, β, r, c, fail, evalF)
end


"""
    runAdam(p0::T1, optAdam::Adam; verbose::Bool=false, seed=123) where T1<:Array{<:AbstractFloat, 1}

    For a function with gradient-(estimate) optAdam.evalGradF and cost function optAdam.computeCost run ADAM 
    using p0 as start guess for optAdam.nIt iterations using the options specified when creating optADAM 
    via createFowardGradientProb. 

    Currently the API is not optimal, and ADAM is only supported for forward-gradient. This will be 
    changed in the future, along with the rest of the API. 
"""
function runAdam(p0::T1, optAdam::Adam; verbose::Bool=false, seed=123) where T1<:Array{<:AbstractFloat, 1}

    Random.seed!(seed)

    nIt = length(optAdam.a)
    optAdam.it = 0
    nParam = length(optAdam.theta)

    # We save nIt+1 values to also keep the first value 
    costVal = Array{Float64, 1}(undef, nIt+1)
    paramVal = Array{Float64, 2}(undef, (nParam, nIt+1))

    paramVal[:, 1] .= p0
    costVal[1] = optAdam.evalF(p0)

    # Set starting values for ADAM 
    optAdam.theta .= p0
    optAdam.thetaOld .= p0
    optAdam.cost = costVal[1]

    if verbose == true
        @showprogress "Running ADAM... " for i in 2:(nIt+1)
            step!(optAdam)
            costVal[i] = optAdam.cost
            paramVal[:, i] .= optAdam.theta
        end
    else
        for i in 2:(nIt+1)
            step!(optAdam)
            costVal[i] = optAdam.cost
            paramVal[:, i] .= optAdam.theta
        end
    end

    iMinCost = argmin(costVal)
    minCost = costVal[iMinCost]
    minParams = paramVal[:, iMinCost]
    
    return minCost, minParams, costVal, paramVal
end


"""
    calcUnbiasedGrad(p::T1, f::Function)::Array{Float64, 1} where T1<:Array{<:AbstractFloat, 1}

    For a function f(p::Vector{AbstractFloat})::AbstractFloat compute an unbiased gradient estimate 
    via g = (∇f·v)v, where · is the dot product and v ~ MV(0, I). (∇f·v) can be computed via a single 
    forward pass using forward automatic differentiation.
"""
function calcUnbiasedGrad(p::T1, f::Function)::Array{Float64, 1} where T1<:Array{<:AbstractFloat, 1}

    v::Array{Float64, 1} = randn(length(p)) # v ~ MV(0, I)
    g = r -> f(p + r*v)
    return ForwardDiff.derivative(g, 0.0) * v
end


"""
    step!(opt::Adam)

    Perform ADAM step for an adam struct. In addition to the classical ADAM 
    in case the loss (cost) cannot be evaluted move back to the parameter 
    vector which produced a non infinite cost, and shrink the step length.
"""
function step!(opt::Adam)

    opt.it += 1

    if opt.it == 1
        cost = opt.evalF(opt.theta)
        if isinf(cost)
            opt.fail = Inf
            println("Error : First function call caused infinite cost")
            return 
        end
    end

    # Compute gradient and update parameter vector theta 
    grad = opt.evalGradF(opt.theta)    
    # ADAM update step in place update for efficency 
    opt.m .= opt.b1 .* opt.m .+ (1 - opt.b1) .* grad
    opt.v .= opt.b2 .* opt.v .+ (1 - opt.b2) .* grad .^ 2
    mhat = opt.m ./ (1 - opt.b1^opt.it)
    vhat = opt.v ./ (1 - opt.b2^opt.it)
    opt.theta .-= opt.β * opt.a[opt.it] * (mhat ./ (sqrt.(vhat) .+ opt.eps))

    # If any parameter is outside of its boundries project back to the box
    moveToBoundaries!(opt)

    # Evaluate cost function 
    cost = opt.evalF(opt.theta)

    # In case cost is inf revert back to old theta and decrease step length 
    if isinf(cost)
        opt.fail += 1
        opt.β *= opt.r 
        opt.theta .= opt.thetaOld
    end

    # If successful update theta and increase step-length up to normal value 
    if !isinf(cost) && all(opt.thetaOld .!== opt.theta)
        opt.β = min(opt.β * opt.c, 1.0)
        opt.fail = 0
        opt.cost = cost
        opt.thetaOld .= opt.theta
    end

    nothing
end


"""
    moveToBoundaries!(opt::Adam)

    For ADAM optimizer move out of bound parameters back into bounds.
"""
function moveToBoundaries!(opt::Adam)
    opt.theta .= min.(max.(opt.theta, opt.lB), opt.uB)
end
