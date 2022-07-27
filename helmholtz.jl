using Gridap
using Gridap.Geometry
using Gridap.Geometry: SkeletonTriangulation, BoundaryTriangulation, CompositeTriangulation
using Printf

include("GridapUtils.jl")
using .GridapUtils: Grad, compile, Div, get_mesh_sizes
using Symbolics 


"""
Solve on Ω the Helmholtz problem -Δ u + u = f0 with Dirichlet or 
Neumann bcs with H^1 conforming Lagrangian elements.
"""
function helmholtz_solver(model, f0, g0, h0, Dtags; pdegree=1)
    @assert pdegree > 0
    reffe = ReferenceFE(lagrangian, Float64, pdegree)
    
    Ntags = [t for t ∈ (5, 6, 7, 8) if t ∉ Dtags]

    δV = TestFESpace(model, reffe; conformity=:H1, dirichlet_tags=Dtags)
    V = TrialFESpace(δV, [g0 for key ∈ Dtags])

    if length(Ntags) == 4
        δV = TestFESpace(model, reffe; conformity=:H1)
        V = TrialFESpace(δV)
    end

    degree = pdegree+1
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 2*(degree-1))

    a(u, v) = ∫( ∇(v)⋅∇(u) )*dΩ + ∫(u*v)*dΩ

    Γ = BoundaryTriangulation(model, tags=Ntags)
    ν = get_normal_vector(Γ)
    dΓ = Measure(Γ, degree+2)
    L(v) = length(Ntags) > 0 ? (∫( v*f0 )*dΩ - ∫(v*(h0⋅ν))*dΓ) : ∫( v*f0 )*dΩ 

    op = AffineFEOperator(a, L, V, δV)

    ls = LUSolver()
    solver = LinearFESolver(ls)

    uh = solve(solver, op)

    return uh
end    

# -- FVM things ---

"""Cell centroid"""
cell_center(x) = sum(x)/length(x)


"""How entities of ω are embedded in entities of Ω"""
function get_embedding_map(Ω::Triangulation, ω::Triangulation)
    trian = ω.dtrian
    if hasproperty(trian, :minus)
        # Skeleton
        @assert hasproperty(trian, :plus)
        mapping = trian.minus.trian.grid.cell_to_parent_cell
        @assert all(mapping .== trian.plus.trian.grid.cell_to_parent_cell)
        return mapping
    end

    trian = ω.dtrian
    # Boundary
    mapping = trian.trian.grid.cell_to_parent_cell
    return mapping
end              


"""CellField over Λ with values of distances between `cell_centers` of entities of Ω 
related to entities of Λ"""
function CellDistance(Ω::Triangulation, Λ)
    Xc = get_cell_coordinates(Ω)
    X = Ω.grid.node_coords

    topology = get_grid_topology(Ω.model)
    cell_type,  = Set(topology.polytopes)
    cell_dim = Dict(QUAD => 2)[cell_type]
    fdim = cell_dim - 1

    f2c = topology.n_m_to_nface_to_mfaces[fdim+1, cell_dim+1]
    f2v = topology.n_m_to_nface_to_mfaces[fdim+1, 1]
    nfacets = length(f2c)
    # We are going to build function with value for each facets
    data = Vector{Float64}(undef, nfacets)
    for f ∈ 1:nfacets
        cells = f2c[f]
        if length(cells) == 1
            c1, = cells
            x1 = cell_center(Xc[c1])
            # The other point will be the edge midpoint
            v1, v2 = f2v[f]
            x2 = 0.5*(X[v1] + X[v2])

            data[f] = norm(x1 - x2)
        else
            c1, c2 = cells
            x1 = cell_center(Xc[c1])
            x2 = cell_center(Xc[c2])
            
            data[f] = norm(x1 - x2) 
        end 
    end
    embedding_map = get_embedding_map(Ω, Λ)
    CellField(data[embedding_map], Λ)
end


"""
Solve on Ω the Helmholtz problem -Δ u + u = f0 with Dirichlet or 
Neumann bcs with finite volume method
"""
function FVM_helmholtz_solver(model, f0, g0, h0, Dtags)
    reffe = ReferenceFE(lagrangian, Float64, 0)
    
    Ntags = [t for t ∈ (5, 6, 7, 8) if t ∉ Dtags]
    δV = TestFESpace(model, reffe; conformity=:L2)
    V = TrialFESpace(δV)

    degree = 2
    # Cell integrals for helmhotlz
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 2*(degree-1))
    # Boundaries for Neumann bcs
    ΓN = BoundaryTriangulation(Ω, tags=Ntags)
    νN = get_normal_vector(ΓN)
    dΓN = Measure(ΓN, degree+2)
    # Boundaries for Dirichlet bcs
    ΓD = BoundaryTriangulation(Ω, tags=Dtags)
    νD = get_normal_vector(ΓD)
    dΓD = Measure(ΓD, degree+2) 
    # Finally for integrating interior facets
    Λ = SkeletonTriangulation(Ω)
    ν = get_normal_vector(Λ)
    dΛ = Measure(Λ, degree+2)

    jumpΛ(arg) = arg.⁺ - arg.⁻

    γΛ = CellDistance(Ω, Λ)
    γΓD = CellDistance(Ω, ΓD)
    
    a(u, v) = ∫(u*v)*dΩ + ∫((1/γΛ)*jumpΛ(u)*jumpΛ(v))*dΛ + ∫((1/γΓD)*u*v)*dΓD

    L(v) = length(Ntags) > 0 ? (∫( v*f0 )*dΩ + ∫((1/γΓD)*u0*v)*dΓD - ∫(v*(h0⋅νN))*dΓN) : (∫( v*f0 )*dΩ + ∫((1/γΓD)*u0*v)*dΓD)

    assemble_matrix(a, V, δV)
    
    op = AffineFEOperator(a, L, V, δV)

    ls = LUSolver()
    solver = LinearFESolver(ls)

    uh = solve(solver, op)

    return uh
end    


# -----

x = Symbolics.variables(:x, 1:2)

# Manufactured solution
u0s = Dict(1 => cos(π*(x[1]*x[2])),
           2 => cos(2*π*x[1])*cos(3*π*x[2]) + 1,
           3 => sin(2*π*x[1])*sin(3*π*x[2]) + 1,
           4 => cos(2*π*(x[1]-x[2])),
           5 => sin(2*π*(x[1]+x[2])))

# FIXME: the FVM parts are fragile BoundaryTriangulation(model) vs BoundaryTriangulation(Ω)
#        cell_center vs cell centroid 
true && begin
    #
    #
    # (3) 6 (4)
    #  7     8
    # (1) 5 (2)
    Dirichlet_tags = [1, 2, 3, 4, 5, 6, 7]
    which = 3
    pdegree = 3 # Polynomial degree of FE space
        
    u0_ = u0s[which]

    flux_ = -Grad(u0_) 
    f0_ = Div(flux_) + u0_

    u0, f0, h0 = (compile(arg, x) for arg in (u0_, f0_, flux_))

    errors, hs, ndofs = [], [], []
    sols = []
    for k ∈ 2:8
        n = 2^(k)

        domain = (0,1,0,1)
        partition = (n, n)
        model = CartesianDiscreteModel(domain, partition)
        # model = simplexify(model)

        # uh = helmholtz_solver(model, f0, u0, h0, Dirichlet_tags; pdegree=pdegree)
        uh = FVM_helmholtz_solver(model, f0, u0, h0, Dirichlet_tags)

        !isempty(sols) && pop!(sols)
        push!(sols, uh)

        Ω = get_triangulation(uh)
        dΩ = Measure(Ω, 2*pdegree)

        e = u0 - uh
        error = sqrt(sum( ∫( e*e )*dΩ ))
        append!(errors, error)
        append!(hs, minimum(get_mesh_sizes(Ω)))
        append!(ndofs, length(get_free_dof_ids(uh.fe_space)))
    end
    uh, = sols

    rates = log.(errors[2:end]./errors[1:end-1])./log.(hs[2:end]./hs[1:end-1])
    rates = [NaN; rates]
    table = hcat(hs, ndofs, errors, rates)
    for row in eachrow(table)
        @printf "h = %.2E dim(V) = %d |u-uh|_1 = %.4E rate = %.2f\n" row...
    end
    writevtk(get_triangulation(uh), "fvm_helmholtz", order=1, cellfields=["uh" => uh])
end