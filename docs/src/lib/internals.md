# Internals

Documentation for `Pathfinder.jl`'s internal functions.

See the [Public Documentation](@ref) section for documentation of the public interface.

## AdvancedHMC integration

The following function is currently intended for interactive use and thus is not part of the public API.

```@docs
Pathfinder.RankUpdateEuclideanMetric
```

## Internal functions

These are meant only for internal use.
However, the docstrings are useful for understanding how Pathfinder works, so they are included here for convenience.

```@autodocs
Modules = [Pathfinder]
Filter = !Base.Fix1(===, Pathfinder.RankUpdateEuclideanMetric)
Public = false
Private = true
```
