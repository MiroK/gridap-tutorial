using Gridap

n = 32
domain = (0,1,0,1)
partition = (n, n)
model = CartesianDiscreteModel(domain, partition)

# Let's have some vector space, 
Velm = ReferenceFE(lagrangian, VectorValue{2, Float64}, 1)
V = TestFESpace(model, Velm)
u = interpolate_everywhere(identity, V)

# Now we want point flux on the Boundary
Γ = BoundaryTriangulation(model)
Qelm = ReferenceFE(lagrangian, Float64, 0)
δQ = TestFESpace(Γ, Qelm)

dΓ = Measure(Γ, 4)
ν = get_normal_vector(Γ)
a(p, q) = ∫(p*q)*dΓ
L(q) = ∫(q*(u⋅ν))*dΓ

op = AffineFEOperator(a, L, δQ, δQ)

ls = LUSolver()
solver = LinearFESolver(ls)
uh = solve(solver, op)


writevtk(get_triangulation(uh), "foo", order=1, cellfields=["uh"=>uh])