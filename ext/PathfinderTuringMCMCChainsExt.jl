module PathfinderTuringMCMCChainsExt

using MCMCChains: MCMCChains
using Pathfinder: Pathfinder
using Turing: Turing

# guards against ambiguity with PathfinderTuringFlexiChainsExt when both are loaded
@static if pkgversion(Turing) < v"0.45"
    Pathfinder._default_turing_chain_type() = MCMCChains.Chains
end

# AbstractMCMC.from_samples for MCMCChains is only defined for the bare unparameterized type,
# not for concrete subtypes, so return MCMCChains.Chains regardless of the concrete type.
Pathfinder._chain_constructor(::MCMCChains.Chains) = MCMCChains.Chains

end  # module
