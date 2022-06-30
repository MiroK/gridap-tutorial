using Gridap
using GridapGmsh

include("GridapUtils.jl")
using .GridapUtils: unit_square_mesh

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