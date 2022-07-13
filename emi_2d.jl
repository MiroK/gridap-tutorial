using Gridap
using GridapGmsh
include("GridapUtils.jl")
using .GridapUtils: split_square_mesh
using .GridapUtils
using Symbolics

"""
TODO
"""
function emi(model, data)
    Ω₁ = Triangulation(model, tags="top_surface")
    Ω₂ = Triangulation(model, tags="bottom_surface")
    Γ = InterfaceTriangulation(Ω₁, Ω₂)
    
    dΩ₁ = Measure(Ω₁, 2)
    dΩ₂ = Measure(Ω₂, 2)
    dΓ = Measure(Γ, 2)
    
    elm = ReferenceFE(lagrangian, Float64, 1)
    
    # 2d 
    dir_tags1 = ["top", "top_left", "top_right", "ul", "ur"]# "iface_left", "iface_right"]
    V1 = TestFESpace(Ω₁, elm; conformity=:H1, dirichlet_tags=dir_tags1)
    U1 = TrialFESpace(V1, data.u0)

    dir_tags2 = ["bottom", "bottom_left", "bottom_right", "ll", "lr"]#, "iface_left", "iface_right"]
    V2 = TestFESpace(Ω₂, elm; conformity=:H1, dirichlet_tags=dir_tags2)
    U2 = TrialFESpace(V2, data.u1)

    Y = MultiFieldFESpace([V1, V2])
    X = MultiFieldFESpace([U1, U2])
    
    κ, κ_ = data.κ0, data.κ1
    
    jumpΓ(u, v) = u.⁺ - v.⁻

    a((u1, u2), (v1, v2)) = (∫(κ*(∇(u1)⋅∇(v1)))*dΩ₁ +  
                             ∫(κ_*(∇(u2)⋅∇(v2)))*dΩ₂ +
                             ∫(jumpΓ(v1, v2)*jumpΓ(u1, u2))*dΓ)

    gu1 = interpolate_everywhere(data.gu, V1)
    gu2 = interpolate_everywhere(data.gu, V2)
    gσ2 = interpolate_everywhere(data.gσ, V2)
    
    # FIXME: SIGNS here?
    L((v1, v2)) = (∫(data.f0*v1)*dΩ₁ + ∫(gu1.⁺*v1.⁺)*dΓ +
                   ∫(data.f1*v2)*dΩ₂ - ∫(gu2.⁻*v2.⁻)*dΓ - ∫(gσ2.⁻*v2.⁻)*dΓ)
    
    op = AffineFEOperator(a, L, X, Y)
    
    ls = LUSolver()
    solver = LinearFESolver(ls)
    
    wh = solve(solver, op)
    uh, uh_ = wh
    
    return uh, uh_
end

# Manufactured solution
x = Symbolics.variables(:x, 1:2)
@variables κ0, κ1

# Top
u0 = sin(π*(x[1]+x[2]))
σ0 = -κ0*Grad(u0)
f0 = Div(σ0)
nΓ0 = Vector{Num}([0, -1])
#
#u1 - u2 = s0*n0 + gu

p0 = Inner(σ0, nΓ0)

# Bottom
u1 = sin(2*π*x[1]*(x[2]-x[1]))
σ1 = -κ1*Grad(u1)
f1 = Div(σ1)
nΓ1 = Vector{Num}([0, 1])
# Interface
gσ = p0 + Inner(σ1, nΓ1)
gu = u0 - u1 - p0
# gu = u0 - u1 - (gs - s1*n1)
#    = u0 - u1 - gs + s1*n1
# u1 - u0 + gu + gs = s1*n1

# Specify for data
κ0_val, κ1_val = 1, 1

u0_exact, u1_exact = [compile(expr, x; κ0=κ0_val, κ1=κ1_val) for expr in (u0, u1)]

f0_exact, f1_exact, gσ_exact, gu_exact = [compile(expr, x; κ0=κ0_val, κ1=κ1_val)
                                          for expr in (f0, f1, gσ, gu)]

data = (
    u0=u0_exact,
    u1=u1_exact,
    #
    κ0=κ0_val,
    κ1=κ1_val,
    #
    f0=f0_exact, f1=f1_exact, gσ=gσ_exact, gu=gu_exact
)

mesh_path, normals = split_square_mesh(0.125, :tri; distance=Inf, offset=0.0)
model = GmshDiscreteModel(mesh_path)

(uh0, uh1) = emi(model, data)

Ω0 = get_triangulation(uh0)
dΩ0 = Measure(Ω0, 5)
e0 = sqrt(sum(∫((uh0 - data.u0)*(uh0 - data.u0))*dΩ0))
writevtk(Ω0, "top", order=1, cellfields=["uh"=>uh0, "e"=>uh0 - data.u0])

Ω1 = get_triangulation(uh1)
dΩ1 = Measure(Ω1, 5)
e1 = sqrt(sum(∫((uh1 - data.u1)*(uh1 - data.u1))*dΩ1))
writevtk(Ω1, "bottom", order=1, cellfields=["uh"=>uh1, "e"=>uh1 - data.u1])

@show (e0, e1)

# TODO
# 1) Document what's up
# 2) Have one MMS with non-trivial flux but such that gσ = 0
# 3) Nitsche method