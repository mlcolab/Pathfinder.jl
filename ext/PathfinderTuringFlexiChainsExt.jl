module PathfinderTuringFlexiChainsExt

using FlexiChains: FlexiChains
using Pathfinder: Pathfinder
using Turing: Turing

# guards against ambiguity with PathfinderTuringMCMCChainsExt when both are loaded
@static if pkgversion(Turing) >= v"0.45"
    Pathfinder._default_turing_chain_type() = FlexiChains.VNChain
end

Pathfinder._chain_type_from_chain(chain::FlexiChains.VNChain) = typeof(chain)

end  # module
