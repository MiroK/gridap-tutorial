# TODO: mms where 1) Dirichlet on 1d and ∂Ω
#                 2) Dirichlet on 1d meet Diriclet on ∂Ω but rest is Neumann
#
# See about dofmap

using Gridap
using GridapGmsh

include("GridapUtils.jl")
using .GridapUtils
using .GridapUtils: Dot, Inner, polygon_mesh, get_mesh_sizes
using Symbolics

#= model_path, normals = GridapUtils.unit_square_mesh(0.125, :tri)
Γ_tag = "bottom"
VΩ_Dtags = ["top", "left", "right", "ll", "ul", "ur", "lr"]
VΓ_Dtags = ["ll", "lr"]
Q_Dtags = ["ll", "lr"]
 =#

function solve_problem(model_path, data)
    model = GmshDiscreteModel(model_path)

    Ω = Triangulation(model)
    Γ = BoundaryTriangulation(model, tags=[data.Γ_tag])

    dΩ = Measure(Ω, 2)
    dΓ = Measure(Γ, 2)

    elm = ReferenceFE(lagrangian, Float64, 1)

    # 2d 
    VΩ = TestFESpace(Ω, elm; conformity=:H1, dirichlet_tags=data.VΩ_Dtags)
    UΩ = TrialFESpace(VΩ, data.u0_exact)
    # 1d
    VΓ = TestFESpace(Γ, elm; conformity=:H1, dirichlet_tags=data.VΓ_Dtags)
    @assert VΓ.ndirichlet == 2
    UΓ = TrialFESpace(VΓ, data.u1_exact)
    # Multiplier
    Q = TestFESpace(Γ, elm; conformity=:H1, dirichlet_tags=data.Q_Dtags)
    @assert Q.ndirichlet == 2
    P = TrialFESpace(Q, data.p_exact)

    Y = MultiFieldFESpace([VΩ, VΓ, Q])
    X = MultiFieldFESpace([UΩ, UΓ, P])

    # FIXME: convergence all dirichlet, Neumann top
    #        what if the Γ edge is tilted <- mms
    #        the curved case
    a((u, u_, p), (v, v_, q)) = ∫((data.kappa*∇(u)⋅∇(v)))*dΩ +  ∫(data.kappa_*(∇(u_)⋅∇(v_)))*dΓ + ∫((v-v_)*p)*dΓ + ∫((u-u_)*q)*dΓ


    L((v, v_, q)) = ∫(data.f0_exact*v)*dΩ + ∫(data.f1_exact*v_)*dΓ + ∫(q*data.g_exact)*dΓ

    #@time A = assemble_matrix(a, X, Y)
    #@time b = assemble_vector(L, Y)

    op = AffineFEOperator(a, L, X, Y)
    ls = LUSolver()
    solver = LinearFESolver(ls)

    wh = solve(solver, op)
    uh, uh_, _ = wh

    return uh, uh_
end

# Setup MMS
polygon = [0 0.; 1 0; 1.1 2; 0 1]
# Here we just want to get the model
model_path, normals = polygon_mesh(polygon, 0.1, :tri)

Γ_tag = "l3_2"
VΩ_Dtags = ["l2_1", "l4_3", "l1_4"]
VΓ_Dtags = ["v2", "v3"]
Q_Dtags = ["v3", "v2"]

x = Symbolics.variables(:x, 1:2)
@variables kappa, kappa_

# Manufactured
u0 = sin(π*(x[1]-x[2]))
σ0 = -kappa*Grad(u0)
f0 = Div(σ0)

nΓ = normals[Γ_tag]
p = Inner(σ0, nΓ)

u1 = sin(2*π*(x[1] + x[2]))  # This is u0 at y = 0

GradΓ(u::Num) = Grad(u) .- Inner(nΓ, Grad(u))*nΓ
GradΓ(u::Vector{Num}) = Grad(u) .- Dot(nΓ, Grad(u))'*nΓ

DivΓ(f::Vector{Num}) = tr(GradΓ(f))
# NOTE: this should be in 1d
f1 = DivΓ(-kappa_*GradΓ(u1)) - p

g = u0 - u1

# Specify for data
kappa_val, kappa__val = 2, 3

u0_exact, u1_exact, p_exact = [compile(expr, x; kappa=kappa_val, kappa_=kappa__val)
                            for expr in (u0, u1, p)]

f0_exact, f1_exact, g_exact = [compile(expr, x; kappa=kappa_val, kappa_=kappa__val)
                            for expr in (f0, f1, g)]

data = (Γ_tag=Γ_tag,
        VΩ_Dtags=VΩ_Dtags,
        VΓ_Dtags=VΓ_Dtags,
        Q_Dtags=Q_Dtags,
        #-----
        kappa=kappa_val,
        kappa_=kappa__val,
        #-----
        u0_exact=u0_exact, u1_exact=u1_exact, p_exact=p_exact,
        f0_exact=f0_exact, f1_exact=f1_exact, g_exact=g_exact
        )

errors, sizes = [], []
for n in 1:5
    scale = 1/2^n
    model_path, _ = polygon_mesh(polygon, scale, :tri)
    uh, uh_ = solve_problem(model_path, data)

    Ω = get_triangulation(uh)
    dΩ = Measure(Ω, 5)
    eu = sqrt(sum(∫((uh-u0_exact)*(uh-data.u0_exact))*dΩ))

    Γ = get_triangulation(uh_)
    dΓ = Measure(Γ, 5)
    eu_ = sqrt(sum(∫((uh_-u1_exact)*(uh_-data.u1_exact))*dΓ))

    push!(errors, [eu, eu_])
    push!(sizes, minimum(get_mesh_sizes(Ω)))

    rates = [-1, -1]
    if length(errors) > 1
        rates = log10.(errors[end]./errors[end-1])./log10(sizes[end]/sizes[end-1])
    end
    @show (errors[end], rates)
end
#writevtk(Ω, "twoD_sol", order=1, cellfields=["uh" => uh, 
#"u" => interpolate_everywhere(data.u0_exact, VΩ)])
#writevtk(Γ, "oneD_sol", order=1, cellfields=["uh_" => uh_,
#"u" => interpolate_everywhere(data.u1_exact, VΓ)]) 


