LogDensityProblems.dimension(lp::PEtab.PEtabPigeonReference) = lp.dim

LogDensityProblems.logdensity(lp::PEtab.PEtabPigeonReference, x) = lp(x)

function PEtab.get_correction(logreference::PEtab.PEtabPigeonReference, x)
    return Bijectors.logabsdetjac(logreference.inference_info.inv_bijectors, x)
end
