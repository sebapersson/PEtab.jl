module PEtabModelingToolkitNeuralNetsExt

import ModelingToolkitNeuralNets
import PEtab

function PEtab._is_neural_network_mtk(id, ::PEtab.ModelSystem)::Bool
    return ModelingToolkitNeuralNets.isneuralnetwork(id)
end

function PEtab._is_neural_network_mtk_ps(id, ::PEtab.ModelSystem)::Bool
    return ModelingToolkitNeuralNets.isneuralnetworkps(id)
end

function PEtab._get_nn_chain_mtk(id, ::PEtab.ModelSystem)
    return ModelingToolkitNeuralNets.get_nn_chain(id)
end

end
