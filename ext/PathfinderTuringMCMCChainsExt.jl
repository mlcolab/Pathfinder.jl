module PathfinderTuringMCMCChainsExt

using MCMCChains: MCMCChains
using Pathfinder: Pathfinder
using Turing: Turing

# guards against ambiguity with PathfinderTuringFlexiChainsExt when both are loaded
if Base.pkgversion(Turing) < v"0.45"
    Pathfinder._default_turing_chain_type() = MCMCChains.Chains
end

end  # module
