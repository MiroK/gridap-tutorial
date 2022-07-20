# TODO: mms where 1) Dirichlet on 1d and ∂Ω
#                 2) Dirichlet on 1d meet Diriclet on ∂Ω but rest is Neumann
#
# See about dofmap

using Gridap
using GridapGmsh

include("GridapUtils.jl")
using .GridapUtils: unit_square_mesh, split_square_mesh


model = GmshDiscreteModel(split_square_mesh(0.1, :tri; distance=Inf, offset=0.0))

function twoDoneD(model, sources, Dbdry_data, Nbdry_data, Ibdry_data)
    Ω = Triangulation(model)
    Ω₁ = Triangulation(model, tags=["top_surface"])
    Ω₂ = Triangulation(model, tags=["bottom_surface"])
    Γ = BoundaryTriangulation(model, tags=["interface"])

    dΩ = Measure(Ω, 2)
    dΩ₁ = Measure(Ω₁, 2)
    dΩ₂ = Measure(Ω₂, 2)
    dΓ = Measure(Γ, 2)

    elm = ReferenceFE(lagrangian, Float64, 1)

# 2d 
VΩ = TestFESpace(Ω, elm; conformity=:H1, dirichlet_tags=["top", "bottom"])
UΩ = TrialFESpace(VΩ, [1, 2])
# 1d
VΓ = TestFESpace(Γ, elm; conformity=:H1, dirichlet_tags=["iface_left"])
UΓ = TrialFESpace(VΓ, 1)
# Multiplier
Q = TestFESpace(Γ, elm; conformity=:H1, dirichlet_tags=["iface_left"])
P = TrialFESpace(Q, [0])

Y = MultiFieldFESpace([VΩ, VΓ, Q])
X = MultiFieldFESpace([UΩ, UΓ, P])

κ = 1E-5
κ_ = 1E2
γ = 1E6

a((u, u_, p), (v, v_, q)) = ∫(κ*(∇(u)⋅∇(v)))*dΩ +  ∫(κ_*(∇(u_)⋅∇(v_)))*dΓ + ∫(γ*(v-v_)*p)*dΓ + ∫(γ*(u-u_)*q)*dΓ


f = 0
f_ = 0
g_ = 1

L((v, v_, q)) = ∫(f*v)*dΩ + ∫(f_*v_)*dΓ + ∫(q*g_)*dΓ

op = AffineFEOperator(a, L, X, Y)

ls = LUSolver()
solver = LinearFESolver(ls)

wh = solve(solver, op)
uh, uh_, _ = wh

writevtk(Ω, "twoD_sol", order=1, cellfields=["uh" => uh])
writevtk(Γ, "oneD_sol", order=1, cellfields=["uh_" => uh_])

u = sqrt(sum(∫(abs(uh))*dΩ))
v = sqrt(sum(∫(abs(uh_))*dΓ))
@show (u, v)
#writevtk(get_triangulation(uh), "stokes_sol", order=1, cellfields=["uh" => uh, "ph" => ph])
end

true && begin
    
end