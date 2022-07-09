using Gridap
using GridapGmsh

include("GridapUtils.jl")
using .GridapUtils: unit_square_mesh, split_square_mesh, compile
using Symbolics

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

"""
Compute permutation of indices of P0 function space of mesh such 
that reordered data can be used in imshow
"""
function imageperm(mesh)
    gdim, _ = eltype(get_grid(mesh).node_coords).parameters
    @assert gdim == 2

    elm = ReferenceFE(lagrangian, Float64, 0)
    V = TestFESpace(mesh, elm; conformity=:L2)

    x = get_free_dof_values(interpolate_everywhere(x -> x[1], V))
    dx = uniform_grid(x)

    y = get_free_dof_values(interpolate_everywhere(x -> x[2], V))
    dy = uniform_grid(y)

    @assert dx == dy || dx == (dy[2], dy[1])
    # Tuples are sorted in "lexicographic which we want"
    idx = sortperm(collect(zip(x, y)))
    reshape(idx, dy)
end

function uniform_grid(x; digits=14)
    x = round.(x; digits=digits)
    # ccc  abc
    # bbb  abc
    # aaa  abc
    indices = Dict{eltype(x), Vector{Integer}}()
    for xi ∈ x
        xi ∉ keys(indices) && setindex!(indices, findall(x -> x == xi, x), xi)
    end
    # We found all
    @assert all(i == v for (i, v) ∈ enumerate(sort(vcat(values(indices)...))))
    # They are equidistant
    diff0 = diff(indices[first(keys(indices))])
    @assert all(norm(diff0 - diff(val)) < 1E-13 for val ∈ values(indices))

    return (length(indices), length(diff0)+1)
end

false && begin
    using Plots    

    model = CartesianDiscreteModel((0, 1, 0, 1), (300, 400))
    Ω = Triangulation(model)

    elm = ReferenceFE(lagrangian, Float64, 0)
    V = TestFESpace(Ω, elm; conformity=:L2)

    f = interpolate_everywhere(x -> sin(2*π*x[1]), V)
    fvals = get_free_dof_values(f)

    perm = imageperm(Ω)
    image = fvals[perm]

    Plots.heatmap(image)
end

true && begin
    mesh_path, normals = unit_square_mesh(0.05, :tri; distance=2)
    
    model = GmshDiscreteModel(mesh_path)
    x = Symbolics.variables(:x, 1:2)
    for (tag, normal) ∈ normals
        Γ = BoundaryTriangulation(model, tags=[tag])
        n = get_normal_vector(Γ)

        error, agreed = Inf, false
        # Wrong up to a sign
        for sign ∈ (1, -1)
            my_normal = compile(sign*normal, x)
            dΓ = Measure(Γ, 1)
            e = ∫((n-my_normal)⋅(n-my_normal))*dΓ
            error = min(error, sqrt(sum(e)))
            # NOTE: this just needs to be a small number of circular arcs
            agreed = agreed || (sign == 1 && error < 1E-4)
        end
        @show (tag, error, agreed)
    end 
end