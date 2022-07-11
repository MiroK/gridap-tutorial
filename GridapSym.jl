using Gridap: VectorValue, TensorValue
using Symbolics

x = Symbolics.variables(:x, 1:2)

Inner(u::Vector{Num}, v::Vector{Num}) = u'*v
Inner(A::Matrix{Num}, v::Vector{Num}) = Dot(A, v)
Inner(v::Vector{Num}, A::Matrix{Num}) = Dot(v, A)

Dot(A::Matrix{Num}, v::Vector{Num}) = A*v
Dot(v::Vector{Num}, A::Matrix{Num}) = A'*v

Dot(A::Matrix{Num}, B::Matrix{Num}) = sum(Dot(a, b) for (a, b) ∈ zip(eachrow(A), eachrow(B)))

Sym(A::Matrix{Num}) = (A + A')/2
Skew(A::Matrix{Num}) = (A - A')/2

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

function Rot(f::Vector{Num}; coords::Vector{Num}=x)
    @assert length(f) == 2
    Div([f[2], -f[1]])
end

function Rot(f::Num; coords::Vector{Num}=x)
    g = Grad(f)
    [g[2], -g[1]]
end

function compile(f::Num, args; kwargs...)
    f_expr = build_function(f, args; expression=Val{true})

    isempty(kwargs) && return eval(f_expr)

    # Now we want to change the function signature
    # At this point it is a tuple :()
    args_ = Vector{Any}()
    # We want to add kwards
    for (k, v) ∈ kwargs
        push!(args_, Expr(:kw, k, v))
    end
    kws = Expr(:parameters, args_...)

    f_expr.args[1] = Expr(:tuple, kws, first(f_expr.args[1].args))

    eval(f_expr)
end

function compile(f::Vector{Num}, args; kwargs...)
    fs = [compile(fi, args; kwargs...) for fi ∈ f]
    g(x; kwargs...) = VectorValue([fi(x; kwargs...) for fi ∈ fs]...)
end

function compile(f::Matrix{Num}, args; kwargs...)
    fs = [compile(fi, args; kwargs...) for fi ∈ f]
    g(x; kwargs...) = TensorValue([fi(x; kwargs...) for fi ∈ fs]...)
end