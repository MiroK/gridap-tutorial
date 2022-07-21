using Gridap
using Gridap.Geometry
using Graphs, SimpleWeightedGraphs


"""
Given a Gridap model where the edges are marked with `graph_tag` use this to 
define a graph and compute a distance between the vertices marked as `start_tag` 
and `end_tag`. 
"""
function graph_distance(model, graph_tag, start_tag, end_tag)
    fb = get_face_labeling(model)
    edge2vertex = model.grid_topology.n_m_to_nface_to_mfaces[2, 1]
    
    graph_edge_indices = get_face_mask(fb, graph_tag, 1)
    graph_edges = edge2vertex[findall(graph_edge_indices)]

    # Weigthed graph of marked edges with edge lengths
    # NOTE: there is some inconsistency in gridap naming so ...
    vertex_x = hasproperty(model.grid, :node_coords) ? model.grid.node_coords : model.grid.node_coordinates

    nvtx = length(unique(vcat(graph_edges...)))
    G = SimpleWeightedGraph(nvtx)
    # NOTE: simple graph won't allow nodes which are outside 1:nvtx (so they are not true labels)
    # This is some indexing issue so we need to renumber the nodes to define the graph ...
    g2l = Dict{Int, Int}()
    counter = 0
    for (gv1, gv2) in graph_edges

        gv1 ∉ keys(g2l) && setindex!(g2l, counter += 1, gv1)
        gv2 ∉ keys(g2l) && setindex!(g2l, counter += 1, gv2)

        lv1, lv2 = g2l[gv1], g2l[gv2]
        l = norm(vertex_x[gv1] - vertex_x[gv2])
        @assert l > 0
        @assert add_edge!(G, lv1, lv2, l)
    end

    l2g = Vector{Int}(undef, length(g2l))
    for (g, l) ∈ g2l
        l2g[l] = g 
    end

    # Get the enfpoints in local numbering
    start_tag = g2l[findfirst(get_face_mask(fb, start_tag, 0))]
    end_tag =  g2l[findfirst(get_face_mask(fb, end_tag, 0))] 
    # Now we query for the path given in terms of vertices
    path = l2g[enumerate_paths(dijkstra_shortest_paths(G, start_tag), end_tag)]
    # NOTE: we map the path back to global to get acces to coords
    distance = sum(norm(vertex_x[path[i]] - vertex_x[path[i+1]]) for i ∈ 1:(length(path)-1))
end

# This is a sanity check
model = CartesianDiscreteModel((0, 1, 0, 1), (100, 100))
@assert abs(graph_distance(model, "boundary", 1, 2)-1) < 1E-5
@assert abs(graph_distance(model, "boundary", 1, 3)-1) < 1E-5
@assert abs(graph_distance(model, "boundary", 1, 4)-2) < 1E-5


# The following shows usage with user defined geometries. Note that 
# the end/start vertices are now refered to via their physical names
false && begin
    include("../GridapUtils.jl")
    using .GridapUtils: split_square_mesh
    using GridapGmsh

    model_path, normals = split_square_mesh(1., offset=0.2, distance=2)
    model = GmshDiscreteModel(model_path)

    graph_distance(model, "interface", "iface_left", "iface_right")
end