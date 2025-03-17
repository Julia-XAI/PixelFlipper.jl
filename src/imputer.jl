"""
    AbstractImputer

Abstract supertype of all imputers. Given an input and indices from a selector, this imputes values at the given index.

Currently, imputing is only supported in the space of floating-point input arrays (`AbstractArrayImputer`). 
In the future, imputers in `RGB` image space could be supported as well.
"""
abstract type AbstractImputer end

# Nice to have in the future:
# abstract type AbstractArrayImputer <: AbstractImputer end
# abstract type AbstractImageImputer <: AbstractImputer end

"""
    ConstantImputer()

Imputer 

Requires a an `AbstractPixelSelector`.
"""
struct ConstantImputer{T,S<:AbstractSelector} <: AbstractImputer
    val::T
    selector::S
end
ConstantImputer(; val = 0.0, selector::AbstractSelector = PixelSelector()) =
    ConstantImputer(val, selector)
