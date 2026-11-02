module PathfinderTuringMCMCChainsExt

using MCMCChains: MCMCChains
using Pathfinder: Pathfinder
using Turing: Turing

# guards against ambiguity with PathfinderTuringFlexiChainsExt when both are loaded
@static if pkgversion(Turing) < v"0.45"
    Pathfinder._default_turing_chain_type() = MCMCChains.Chains
end

# AbstractMCMC.from_samples for MCMCChains is only defined for the bare unparameterized
# Chains type, not for concrete subtypes.
Pathfinder._chain_type_from_chain(::MCMCChains.Chains) = MCMCChains.Chains

end  # module
