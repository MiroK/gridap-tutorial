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
# uᵢ is prescribed on the remaining boundaries ∂Ωᵢ∖ Γ, except for 
# the top edge where we fix the flux σ₁⋅ν₁.
using Printf
using Symbolics

using Gridap
using GridapGmsh
include("GridapUtils.jl")
using .GridapUtils: split_square_mesh, get_mesh_sizes
using .GridapUtils


"""
The solution approach to EMI equations where ϵ >= 0 is assumed. We consider 
mixed form for one of the Poisson equations while keeping primal in the other. 
We end up with a perturbed saddle point problem. The flux condition on σ₁⋅ν₁ 
is here enforced by Nitsche method. 
"""
function emi(model, parameters, data)
    # NOTE: the tags here have a specific model/tagging in mind. It
    # should be the split square geoemetry
    Ω₁ = Triangulation(model, tags="top_surface")
    Ω₂ = Triangulation(model, tags="bottom_surface")
    # Gridp normal will point from 1(+) to 2(-)
    Γ = InterfaceTriangulation(Ω₁, Ω₂)
    nΓ = get_normal_vector(Γ)

    dΩ₁ = Measure(Ω₁, 2)
    dΩ₂ = Measure(Ω₂, 2)
    dΓ = Measure(Γ, 2)
    
    # In the first domain we will have mixed Darcy
    S1elm = ReferenceFE(raviart_thomas, Float64, 0)
    V1elm = ReferenceFE(lagrangian, Float64, 0)
    
    # We consider pressure bcs which will be here enforced weakly ...
    pressure_tags1 = ["top_left", "top_right"]
    # ...and flux on top
    flux_tags1 = ["top"]
    
    Γ1 = BoundaryTriangulation(Ω₁, tags=pressure_tags1)
    # 
    δS1 = TestFESpace(Ω₁, S1elm; conformity=:Hdiv)
    δV1 = TestFESpace(Ω₁, V1elm; conformity=:L2)
    nΓ1 = get_normal_vector(Γ1)
    dΓ1 = Measure(Γ1, 2)
    
    S1, V1 = TrialFESpace(δS1), TrialFESpace(δV1)
    
    # In the second domain we will have primal poisson (with strong bcs)
    V2elm = ReferenceFE(lagrangian, Float64, 1)
    dir_tags2 = ["bottom", "bottom_left", "bottom_right", "ll", "lr"]
    δV2 = TestFESpace(Ω₂, V2elm; conformity=:H1, dirichlet_tags=dir_tags2)
    V2 = TrialFESpace(δV2, data.u1)

    δX = MultiFieldFESpace([δS1, δV1, δV2])
    X = MultiFieldFESpace([S1, V1, V2])
    
    κ, κ_ = parameters.κ0, parameters.κ1
    # NOTE: CellFields seem necessary to make the integration work proper 
    # on Γ. Otheriwse restrictions are needed
    ϵ = CellField(parameters.ϵ, Γ)
    
    ΓN = BoundaryTriangulation(Ω₁, tags=flux_tags1)
    nΓN = get_normal_vector(ΓN)
    dΓN = Measure(ΓN, 2)
    # Nitsche auxiliary vars
    γΓN = 20
    hΓN = CellField(lazy_map(h->h, get_array(∫(1)*dΓN)), ΓN) 

    a((s1, u1, u2), (t1, v1, v2)) = (
        ∫((1/κ)*(s1⋅t1))*dΩ₁ + ∫(ϵ*(s1.⁺⋅nΓ.⁺)*(t1.⁺⋅nΓ.⁺))*dΓ - ∫(u1*(∇⋅t1))*dΩ₁ + ∫(u2.⁻*(t1.⁺⋅nΓ.⁺))*dΓ 
        - ∫(v1*(∇⋅s1))*dΩ₁
        + ∫(v2.⁻*(s1.⁺⋅nΓ.⁺))*dΓ                                                  - ∫(κ_*(∇(u2)⋅∇(v2)))*dΩ₂
        # Now the Nitsche terms
        + ∫(u1*(t1⋅nΓN))*dΓN + ∫(v1*(s1⋅nΓN))*dΓN + ∫((γΓN/hΓN)*(t1⋅nΓN)*(s1⋅nΓN))*dΓN
    )

    gu = CellField(data.gu, Γ)
    gσ = CellField(data.gσ, Γ)
    
    L((t1, v1, v2)) = (
         ∫(-gu*(t1.⁺⋅nΓ.⁺))*dΓ - ∫(data.u0*(t1⋅nΓ1))*dΓ1 
        -∫(data.f0*v1)*dΩ₁ 
        -∫(data.f1*v2)*dΩ₂ + ∫(gσ*v2.⁻)*dΓ
        # Now the Nitsche terms
        + ∫(v1*(data.σ0⋅nΓN))*dΓN + ∫((γΓN/hΓN)*(t1⋅nΓN)*(data.σ0⋅nΓN))*dΓN
    )
    
    op = AffineFEOperator(a, L, X, δX)
    
    ls = LUSolver()
    solver = LinearFESolver(ls)
    
    wh = solve(solver, op)
    s1h, uh, uh_ = wh
    
    return s1h, uh, uh_
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

σ0_exact, u0_exact, u1_exact = [compile(expr, x; params...) for expr in (σ0, u0, u1)]

f0_exact, f1_exact, gσ_exact, gu_exact = [compile(expr, x; params...)
                                          for expr in (f0, f1, gσ, gu)]

data = (σ0=σ0_exact, u0=u0_exact, u1=u1_exact,
        f0=f0_exact, f1=f1_exact, gσ=gσ_exact, gu=gu_exact)
# As named tuple
parameters = (; params...)

# Convergence check
x_true = [σ0_exact, u0_exact, u1_exact]

sizes, errors = [], []
for n ∈ 1:4
    scale = 1/2^n

    mesh_path, normals = split_square_mesh(scale, :tri)
    model = GmshDiscreteModel(mesh_path)

    xh = emi(model, parameters, data)
    errors_n = Vector{Float64}(undef, length(xh))
    sizes_n, dofs_n = similar(errors_n), similar(errors_n)
    for (i, xih) ∈ enumerate(xh)
        Ωi = get_triangulation(xih) 
        dΩi = Measure(Ωi, 5)

        errors_n[i] = sqrt(sum(∫((xih - x_true[i])⋅(xih - x_true[i]))*dΩi))
        sizes_n[i] = minimum(get_mesh_sizes(Ωi))
        dofs_n[i] = length(get_free_dof_values(xih))
    end
    @show errors
    # ---

    push!(sizes, sizes_n)
    push!(errors, errors_n)

    rates = length(errors) == 1 ? [-1, -1] : log.(errors[end]./errors[end-1])./log.(sizes[end]./sizes[end-1])

    table = zip(sizes[end], dofs_n, errors[end], rates)
    for (i, row) in enumerate(table)
        @printf "\x1B[35m h = %.2E dim(V) = %d | |u-uh| = %.2E rate = %.2f\n\033[0m" row...
    end
end