# Internals

Documentation for `Pathfinder.jl`'s internal functions.

See the [Public Documentation](@ref) section for documentation of the public interface.

## AdvancedHMC integration

```@docs
AdvancedHMC.RankUpdateEuclideanMetric(::Pathfinder.WoodburyPDMat)
```

## Internal functions

These are meant only for internal use.
However, the docstrings are useful for understanding how Pathfinder works, so they are included here for convenience.

```@autodocs
Modules = [Pathfinder]
Public = false
Private = true
```
