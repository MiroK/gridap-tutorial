# The EMI equations describe a surface coupling of two Poisson problems
#
# in Ω₁ we have: 
#                 ∇⋅σ₁ = f₁ where σ₁ = -κ₁∇u₁
#
# in Ω₂ we have: 
#                 ∇⋅σ₂ = f₂ where σ₂ = -κ₂∇u₂
#
# on the interface Γ:=∂Ω₁ ∩ ∂Ω₂ we have:
#
#                 u₁ - u₂ = ϵ σ₁⋅ν₁ + gu          (1)
#               σ₁⋅ν₁ + σ₂⋅ν₂ = gσ
#
# Here νᵢ is the outer normal of Ωᵢ. In physical applications 
# gσ = 0. The parameter ϵ is non-negative. Finally we use that 
# uᵢ is prescribed on the remaining boundaries ∂Ωᵢ∖ Γ.
using Printf
using Symbolics

using Gridap
using GridapGmsh
include("GridapUtils.jl")
using .GridapUtils: split_square_mesh, get_mesh_sizes
using .GridapUtils


# FIXME: This doesn't work - make a MWE and report to gridap
"""
The solution approach to EMI equations where ϵ= > 0 is assumed and 
we use a Lagrange multriplier to enforce the condition (1).
"""
function emi(model, parameters, data)
    # NOTE: the tags here have a specific model/tagging in mind. It
    # should be the split square geoemetry
    Ω₁ = Triangulation(model, tags="top_surface")
    Ω₂ = Triangulation(model, tags="bottom_surface")
    # Gridp normal will point from 1(+) to 2(-)
    Γ = InterfaceTriangulation(Ω₁, Ω₂)
    
    dΩ₁ = Measure(Ω₁, 2)
    dΩ₂ = Measure(Ω₂, 2)
    dΓ = Measure(Γ, 2)
    
    elm = ReferenceFE(lagrangian, Float64, 1)
    
    # Boundaries for strong Dirichlet bcs
    dir_tags1 = ["top", "top_left", "top_right", "ul", "ur"]
    V1 = TestFESpace(Ω₁, elm; conformity=:H1, dirichlet_tags=dir_tags1)
    U1 = TrialFESpace(V1, data.u0)

    dir_tags2 = ["bottom", "bottom_left", "bottom_right", "ll", "lr"]
    V2 = TestFESpace(Ω₂, elm; conformity=:H1, dirichlet_tags=dir_tags2)
    U2 = TrialFESpace(V2, data.u1)

    # The multiplier
    Q = TestFESpace(Γ, elm; conformity=:H1, dirichlet_tags=["iface_left", "iface_right"])
    P = TrialFESpace(Q, 0)

    Y = MultiFieldFESpace([V1, V2, Q])
    X = MultiFieldFESpace([U1, U2, P])
    
    κ, κ_ = parameters.κ0, parameters.κ1
    # NOTE: CellFields seem necessary to make the integration work proper 
    # on Γ. Otheriwse restrictions are needed
    ϵ = CellField(parameters.ϵ, Γ)
    
    jumpΓ(u, v) = u.⁺ - v.⁻

    a((u1, u2, p), (v1, v2, q)) = (
        # Bulk
        ∫(κ*(∇(u1)⋅∇(v1)))*dΩ₁ + ∫(κ_*(∇(u2)⋅∇(v2)))*dΩ₂
        # Coupling
        # + ∫(p*jumpΓ(v1, v2))*dΓ + ∫(q*jumpΓ(u1, u2))*dΓ 
        # Plus/Minus restriction does not help here
        - ∫(ϵ*p*q)*dΓ
    )

    gu = CellField(data.gu, Γ)
    gσ = CellField(data.gσ, Γ)
    
    L((v1, v2, q)) = (
        # Bulk
        ∫(data.f0*v1)*dΩ₁ + ∫(data.f1*v2)*dΩ₂
        # Coupling
        #- ∫(gσ*v2.⁻)*dΓ + ∫(gu*q)*dΓ
    )
    
    op = AffineFEOperator(a, L, X, Y)
    
    ls = LUSolver()
    solver = LinearFESolver(ls)
    
    wh = solve(solver, op)
    uh, uh_, _ = wh
    
    return uh, uh_
end

# Manufactured solution have in mind the mesh 
# where the interca will be the line y = 0.5
x = Symbolics.variables(:x, 1:2)
# NOTE: I switched to 0 based indexing here
@variables κ0, κ1, ϵ

# Top
u0 = sin(π*(x[1]+x[2]))
σ0 = -κ0*Grad(u0)
f0 = Div(σ0)
# The domain is above y = 0.5 so normal is
nΓ0 = Vector{Num}([0, -1])
# This would be the Lagrange multiplier
p0 = Inner(σ0, nΓ0)

# Bottom
u1 = sin(2*π*x[1]*(x[2]-x[1]))
σ1 = -κ1*Grad(u1)
f1 = Div(σ1)
# We look up
nΓ1 = Vector{Num}([0, 1])
# Interface coupling
gσ = p0 + Inner(σ1, nΓ1)
gu = u0 - u1 - ϵ*p0

# Specify for data
params = Dict(:κ0 => 3, :κ1 => 2, :ϵ => 0.5)

u0_exact, u1_exact = [compile(expr, x; params...) for expr in (u0, u1)]

f0_exact, f1_exact, gσ_exact, gu_exact = [compile(expr, x; params...)
                                          for expr in (f0, f1, gσ, gu)]

data = (u0=u0_exact, u1=u1_exact,
        f0=f0_exact, f1=f1_exact, gσ=gσ_exact, gu=gu_exact)
# As named tuple
parameters = (; params...)

sizes, errors = [], []
for n ∈ 1:1
    scale = 1/2^n

    mesh_path, normals = split_square_mesh(scale, :tri)
    model = GmshDiscreteModel(mesh_path)

    (uh0, uh1) = emi(model, parameters, data)

    Ω0 = get_triangulation(uh0)
    dΩ0 = Measure(Ω0, 5)
    e0 = sqrt(sum(∫((uh0 - data.u0)*(uh0 - data.u0))*dΩ0))

    Ω1 = get_triangulation(uh1)
    dΩ1 = Measure(Ω1, 5)
    e1 = sqrt(sum(∫((uh1 - data.u1)*(uh1 - data.u1))*dΩ1))
    "\033[0m"
    @show (e0, e1)

    push!(sizes, [minimum(get_mesh_sizes(Ω0)), minimum(get_mesh_sizes(Ω1))])
    push!(errors, [e0, e1])

    rates = length(errors) == 1 ? [-1, -1] : log.(errors[end]./errors[end-1])./log.(sizes[end]./sizes[end-1])

    table = zip(sizes[end], errors[end], rates)
    for (i, row) in enumerate(table)
        @printf "\x1B[35m h = %.2E | |u-uh| = %.2E rate = %.2f\n\033[0m" row...
    end
end