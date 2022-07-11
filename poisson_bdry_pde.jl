# TODO: mms where 1) Dirichlet on 1d and ∂Ω
#                 2) Dirichlet on 1d meet Diriclet on ∂Ω but rest is Neumann
#
# See about dofmap

using Gridap
using GridapGmsh

include("GridapUtils.jl")
using .GridapUtils
using .GridapUtils: Dot, Inner
using Symbolics

x = Symbolics.variables(:x, 1:2)
@variables kappa, kappa_

# Manufactured
u0 = sin(π*(x[1]-x[2]))
σ0 = -kappa*Grad(u0)
f0 = Div(σ0)

nΓ = Vector{Num}([0, -1])
p = Inner(σ0, nΓ)

u1 = sin(2*π*(x[1] + x[2]))  # This is u0 at y = 0

GradΓ(u::Num) = Grad(u) .- Inner(nΓ, Grad(u))*nΓ
GradΓ(u::Vector{Num}) = Grad(u) .- Dot(nΓ, Grad(u))'*nΓ

DivΓ(f::Vector{Num}) = tr(GradΓ(f))
# NOTE: this should be in 1d
f1 = DivΓ(-kappa_*GradΓ(u1)) - p

g = u0 - u1

# Specify for data
kappa_val, kappa__val = 1, 2

u0_exact, u1_exact, p_exact = [compile(expr, x; kappa=kappa_val, kappa_=kappa__val)
                               for expr in (u0, u1, p)]

f0_exact, f1_exact, g_exact = [compile(expr, x; kappa=kappa_val, kappa_=kappa__val)
                               for expr in (f0, f1, g)]

# Solve for resolution
model_path, normals = GridapUtils.unit_square_mesh(0.125, :tri)
model = GmshDiscreteModel(model_path)

Ω = Triangulation(model)
Γ = BoundaryTriangulation(model, tags=["bottom"])

dΩ = Measure(Ω, 2)
dΓ = Measure(Γ, 2)

elm = ReferenceFE(lagrangian, Float64, 1)

# 2d 
VΩ = TestFESpace(Ω, elm; conformity=:H1, dirichlet_tags=["top", "left", "right", "ll", "ul", "ur", "lr"])
UΩ = TrialFESpace(VΩ, u0_exact)
# 1d
VΓ = TestFESpace(Γ, elm; conformity=:H1, dirichlet_tags=["ll", "lr"])
UΓ = TrialFESpace(VΓ, u1_exact)
# Multiplier
Q = TestFESpace(Γ, elm; conformity=:H1, dirichlet_tags=["ll", "lr"])
P = TrialFESpace(Q, p_exact)

Y = MultiFieldFESpace([VΩ, VΓ, Q])
X = MultiFieldFESpace([UΩ, UΓ, P])

# FIXME: convergence all dirichlet, Neumann top
#        what if the Γ edge is tilted <- mms
#        the curved case
a((u, u_, p), (v, v_, q)) = ∫((∇(u)⋅∇(v)))*dΩ +  ∫(kappa__val*(∇(u_)⋅∇(v_)))*dΓ + ∫((v-v_)*p)*dΓ + ∫((u-u_)*q)*dΓ


L((v, v_, q)) = ∫(f0_exact*v)*dΩ + ∫(f1_exact*v_)*dΓ + ∫(q*g_exact)*dΓ

#@time A = assemble_matrix(a, X, Y)
#@time b = assemble_vector(L, Y)

op = AffineFEOperator(a, L, X, Y)
ls = LUSolver()
solver = LinearFESolver(ls)

wh = solve(solver, op)
uh, uh_, _ = wh

writevtk(Ω, "twoD_sol", order=1, cellfields=["uh" => uh, 
                                             "u" => interpolate_everywhere(u0_exact, VΩ)])
writevtk(Γ, "oneD_sol", order=1, cellfields=["uh_" => uh_,
                                             "u" => interpolate_everywhere(u1_exact, VΓ)]) 


@show sqrt(sum(∫((uh-u0_exact)*(uh-u0_exact))*dΩ))
@show sqrt(sum(∫((uh_-u1_exact)*(uh_-u1_exact))*dΓ))
#v = sqrt(sum(∫(abs(uh_))*dΓ))
#@show (u, v)
##writevtk(get_triangulation(uh), "stokes_sol", order=1, cellfields=["uh" => uh, "ph" => ph])
#end
#
#true && begin
#    
#end
#end