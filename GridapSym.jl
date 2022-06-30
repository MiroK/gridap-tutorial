using Gridap: VectorValue, TensorValue
using LinearAlgebra: tr 
using Symbolics
import LinearAlgebra: ⋅

x = Symbolics.variables(:x, 1:2)

⋅(u::Vector{Num}, v::Vector{Num}) = xx'*xx
⋅(A::Matrix{Num}, v::Vector{Num}) = A*v
⋅(v::Vector{Num}, A::Matrix{Num}) = A'*v

⊙(A::Matrix{Num}, B::Matrix{Num}) = sum((a⋅b) for (a, b) ∈ zip(eachrow(A), eachrow(B)))

sym(A::Matrix{Num}) = (A + A')/2
skew(A::Matrix{Num}) = (A - A')/2

function Grad(f::Num; coords::Vector{Num}=x)
    # Here we assume that the highest x_i given represents the geometric
    Symbolics.gradient(f, coords)
end

function Grad(f::Vector{Num}; coords::Vector{Num}=x)
    # Here we assume that the highest x_i given represents the geometric
    Symbolics.jacobian(f, coords)
end

function Div(f::Vector{Num}; coords::Vector{Num}=x)
    tr(Grad(f, coords=coords))
end

function Div(f::Matrix{Num}; coords::Vector{Num}=x)
    [Div(f[row, :], coords=coords) for row ∈ 1:first(size(f))]
end

function Curl(f::Vector{Num}; coords::Vector{Num}=x)
    [Symbolics.derivative(f[3], coords[2])-Symbolics.derivative(f[2], coords[3]),
     Symbolics.derivative(f[1], coords[3])-Symbolics.derivative(f[3], coords[1]),
     Symbolics.derivative(f[2], coords[1])-Symbolics.derivative(f[1], coords[2])]
end

function compile(f::Num, args)
    build_function(f, args; expression=Val{false})
end

function compile(f::Vector{Num}, args)
    fs = [build_function(f[i], args; expression=Val{false})
          for i ∈ eachindex(f)] 
    g(x) = VectorValue([fs[i](x) for i ∈ eachindex(fs)]...)
end

function compile(f::Matrix{Num}, args)
    fs = [build_function(f[i], args; expression=Val{false})
          for i ∈ eachindex(f)] 
    g(x) = TensorValue([fs[i](x) for i ∈ eachindex(fs)]...)
end