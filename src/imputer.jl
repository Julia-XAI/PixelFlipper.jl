"""
    AbstractImputer

Abstract supertype of all imputers. Given an input and indices from a selector, this imputes values of an array at the given index.
"""
abstract type AbstractImputer end

"""
    ConstantImputer(value)

Imputes a constant value into a given array. Defaults to `0.0f0`. 
"""
@kwdef struct ConstantImputer{T} <: AbstractImputer
    val::T = 0.0f0
end

function impute!(x, I::CartesianIndices, imp::ConstantImputer)
    x[I] .= imp.val
    return x
end
