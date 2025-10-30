# Public Documentation

Documentation for `Pathfinder.jl`'s public interface.

See the [Internals](@ref) section for documentation of internal functions.

Pathfinder can be run in two primary modes: single- and multi-path.


## Single-path Pathfinder

```@autodocs
Modules = [Pathfinder]
Pages = ["singlepath.jl"]
Order = [:function, :type]
Public = true
Private = false
```

## Multi-path Pathfinder

```@autodocs
Modules = [Pathfinder]
Pages = ["multipath.jl"]
Order = [:function, :type]
Public = true
Private = false
```

## Turing integration

The above functions have special overloads for supporting Turing models.

```@docs
Pathfinder.pathfinder(::DynamicPPL.Model; kwargs...)
Pathfinder.multipathfinder(::DynamicPPL.Model, ::Int; kwargs...)
```
