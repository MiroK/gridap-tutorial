using Gridap
using LightGraphs

"""
Let V be a TestFESpace on a simple open curve. We compute a function 
with values of archlength coordinate.
"""
function arclength_coordinate(V; tol=1E-10)
    # FIXME:
    # Check that we are in 2d and have some scalar elements with dofs being 
    # the point evaluations. Also, here we assume that V = V(Γ) where Γ is an 
    # from InterfaceTriangulation
    Γ = get_triangulation(V)
    # The idea is to build a graph and walk it
    cell_idx = Γ.plus.trian.tface_to_mface
    parent_topology = Γ.plus.trian.model.grid_topology.n_m_to_nface_to_mfaces[2, 1]
    # Encode the cells of \Gamma in terms of parent vertices 
    cell_vtx = parent_topology[cell_idx, :]
    l2g = unique(hcat(cell_vtx...))  # In parent numbering
    # We want to build the graph in terms of local numbering
    g2l = Dict(map(reverse, enumerate(l2g)))
    G = SimpleGraph(length(l2g))
    for (g0, g1) ∈ cell_vtx
        add_edge!(G, g2l[g0], g2l[g1])
    end

    # What's the degree of each vertex?
    degrees = degree(G)
    # Check that this is a simple open curve
    @assert all(1 .<= degrees .<= 2)
    # Our path will march between two vertices connected only to single cell each
    start, stop = findall(isequal(1), degrees)
    path = enumerate_paths(dijkstra_shortest_paths(G, start), stop)
    
    cell_vtx_l = Dict{Tuple{Int, Int}, Int}()
    # Now we want to walk in terms of cells so we build a lookup
    for (ci, (g0, g1)) ∈ enumerate(cell_vtx)
        l0, l1 = g2l[g0], g2l[g1]
        # Sort the key
        if l0 < l1
            cell_vtx_l[(l0, l1)] = ci
        else
            cell_vtx_l[(l1, l0)] = ci
        end
    end

    # While we walk we want to build the arclength
    node_x = Γ.plus.trian.grid.parent.node_coordinates[l2g]
    # The idea is to insert nodes based on their distance; so let's get their position
    # Here we assume 2D
    dofs_x = get_free_dof_values(interpolate_everywhere(x -> x[1], V))
    dofs_y = get_free_dof_values(interpolate_everywhere(x -> x[2], V))
    dm = get_cell_dof_ids(V)
    # The vector of dof values we are building is 
    dist = similar(dofs_x)

    distance = 0
    for i ∈ 1:(length(path)-1)
        l0, l1 = (path[i], path[i+1])
        key = l0 < l1 ? (l0, l1) : (l1, l0)
        # Fing the cells with these two vertices
        ci = cell_vtx_l[key]

        # Start and end coord
        x0, y0 = node_x[l0]
        x1, y1 = node_x[l1]
        edge_length = sqrt((x0 - x1)^2 + (y0 - y1)^2)
        
        cell_dofs = dm[ci]
        # For the dofs to be set get their distance from l0
        dofs_dist = sqrt.((dofs_x[cell_dofs] .- x0).^2 .+ (dofs_y[cell_dofs] .- y0).^2)
        @assert all(-tol .< dofs_dist .< edge_length+tol)
        # The arclength is based on the cumsum
        dist[cell_dofs] = dofs_dist .+ distance
        # For the next round we start with the l1 vertex
        distance += edge_length
    end

    return FEFunction(V, dist)
end


using GridapGmsh

include("../GridapUtils.jl")
using .GridapUtils: split_square_mesh


model_path, _ = split_square_mesh(0.2)#; offset=0.2, distance=2)

model = GmshDiscreteModel(model_path)

Ω0 = Triangulation(model, tags=["top_surface"])
Ω1 = Triangulation(model, tags=["bottom_surface"])
Γ = InterfaceTriangulation(Ω0, Ω1)

# For representing flux
Qelm = ReferenceFE(lagrangian, Float64, 1)
δQ = TestFESpace(Γ, Qelm)

al = arclength_coordinate(δQ)

dΓ = Measure(Γ, 4)
# The test here is that the tangent should be orthogonal
ν = get_normal_vector(Γ)

τ = TensorValue(0., -1, 1, 0)⋅∇(al)

e0 = sum(∫(τ⋅ν.⁻)*dΓ)

writevtk(Γ, "bar", order=1, cellfields=["ddd" => τ, "al" => al])

@show (e0, )

