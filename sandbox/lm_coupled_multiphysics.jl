using Gridap

model = DiscreteModelFromFile("elasticFlag.json")

Ω = Triangulation(model)
Ωs = Triangulation(model,tags="solid")
Ωf = Triangulation(model,tags="fluid")

Γ = InterfaceTriangulation(Ωs, Ωf)
nΓ = get_normal_vector(Γ)

dΓ = Measure(Γ, 4)

δVs = TestFESpace(Ωs, ReferenceFE(lagrangian, Float64, 1); conformity=:H1)
δVf = TestFESpace(Ωf, ReferenceFE(lagrangian, Float64, 1); conformity=:H1)
δQ = TestFESpace(Γ, ReferenceFE(lagrangian, Float64, 1); conformity=:H1)
Us, Uf, Q = TrialFESpace(δVs), TrialFESpace(δVf), TrialFESpace(δQ)

# This works
δX = MultiFieldFESpace([δVs, δVf])
X = MultiFieldFESpace([Us, Uf])

a((us, uf), (vs, vf)) = ∫(us.⁺*vs.⁺)*dΓ
A = assemble_matrix(a, X, δX)
@assert norm(A) > 0

# Raises
δZ = MultiFieldFESpace([δVs, δVf, δQ])
Z = MultiFieldFESpace([Us, Uf, Q])

c((us, uf, p), (vs, vf, q)) = ∫(us.⁺*vs.⁺)*dΓ
C = assemble_matrix(c, Z, δZ)