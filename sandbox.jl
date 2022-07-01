using Gridap
using GridapGmsh

include("GridapUtils.jl")
using .GridapUtils: unit_square_mesh, split_square_mesh

"""
Can we have boundary of a boundary?
"""
function boundary_of_boundary()
    model = GmshDiscreteModel(unit_square_mesh(0.2))

    Γ = BoundaryTriangulation(model, tags=["left"])
    dΓ = Measure(Γ, 2)
    elm = ReferenceFE(lagrangian, Float64, 1)
    V = TestFESpace(Γ, elm; conformity=:H1)
    W = TestFESpace(Γ, elm; dirichlet_tags=["ll", "ul"])
    # At least we can set bcs it seems
    @assert W.nfree + 2 == V.nfree

    δ = DiracDelta{0}(model, tags=["ll"])

    a(u, v) = ∫(u*v)*dΓ
    L(v) = δ(v)

    op = AffineFEOperator(a, L, V, V)
end


function get_tangent_component()
    model = GmshDiscreteModel(unit_square_mesh(0.2))

    Γ = BoundaryTriangulation(model)
    dΓ = Measure(Γ, 2)
    nΓ = get_normal_vector(Γ)

    elm = ReferenceFE(lagrangian, Float64, 0)
    V = TestFESpace(Γ, elm; conformity=:L2)
    
    f(x) = VectorValue(x[1], x[2])
    tangent(f, n) = f⋅(TensorValue(0, 1, -1, 0)⋅n)
    b(v) = ∫(tangent(f, nΓ)*v)*dΓ

    assemble_vector(b, V)
end


function piecewise_integration()
    model = GmshDiscreteModel(split_square_mesh(0.2))

    Ω = Triangulation(model)
    Ω₁ = Triangulation(model, tags=["top_surface"])
    Ω₂ = Triangulation(model, tags=["bottom_surface"])

    dΩ₁ = Measure(Ω₁, 2)
    dΩ₂ = Measure(Ω₂, 2)

    elm = ReferenceFE(lagrangian, Float64, 0)
    V = TestFESpace(Ω, elm; conformity=:L2)
    
    f₁(x) = 1
    f₂(x) = 2

    K = get_cell_measure(Ω)
    K = CellField(lazy_map(K -> K, K), Ω)
    L(v) = ∫((f₁/K)*v)*dΩ₁ + ∫((f₂/K)*v)*dΩ₂

    b = assemble_vector(L, V)

    uh = FEFunction(V, b)
end


function interface_integration()
    model = GmshDiscreteModel(split_square_mesh(0.2; distance=2))

    Ω = Triangulation(model)
    Γ = BoundaryTriangulation(model, tags=["interface"])
    dΓ = Measure(Γ, 1)

    elm = ReferenceFE(lagrangian, Float64, 1)
    V = TestFESpace(Ω, elm; conformity=:H1)
    
    f(x) = VectorValue(x[1], x[2])
    ν = get_normal_vector(Γ)
    L(v) = ∫(v*f⋅ν)*dΓ

    b = assemble_vector(L, V)
end
