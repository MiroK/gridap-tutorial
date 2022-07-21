using LinearAlgebra
using Gridap.Geometry: CompositeTriangulation
using Gridap
using Graphs, SimpleWeightedGraphs


"""Segment mesh in Xd"""
struct SegmentMesh{T}
    # NOTE: colums of the matrix are the vertices
    vertices::Matrix{T}
    topology::Vector{Vector{Int}}
end


"""|A, B|"""
struct Segment{T, S}
    A::Vector{T}
    B::Vector{S}
    τ::Vector{Float64}
    l::Float64    
end


"""Precompute some things for intersection computing"""
function Segment(A, B)
    τ = B .- A
    l = norm(τ, 2)
    Segment(A, B, τ, l)
end


"""point to |A, B| distance"""
function segment_distance(segment::Segment, point; tol=1E-10)
    A, B = segment.A, segment.B
    τ, l = segment.τ, segment.l
    # We look for point X on the line A-B where X-P is orthogonal
    # to the line tangent
    s = dot(point .- A, τ)/l^2

    d = min(norm(point .- A), norm(point .- B))
    # If the point is not interior to the segment we retur distance to endpoints
    (s > 1 + tol || s < -tol) && return d
    # Otherwise it is the distance
    X = A .+ τ*s
    norm(point .- X)
end


"""Collection of segments"""
struct Curve
    segments::Vector{Segment}
end


"""Build the curve from segments"""
function Curve(mesh::SegmentMesh; leafsize=10)
    vertices = mesh.vertices
    cell2vertex = mesh.topology

    segments = Vector{Segment}()
    resize!(segments, length(cell2vertex))
    for (i, cell) ∈ enumerate(cell2vertex)
        segments[i] = Segment(vertices[:, cell[1]], vertices[:, cell[2]])
    end

    Curve(segments)
end


"""Compute distance of point to curve (by bruteforce :|)"""
# NOTE: this is the least specific type aimed at vectors, views atc
function curve_distance(Γ::Curve, point; tol=1E-10)
    distance = Inf
    for seg ∈ Γ.segments
        distance = min(distance, segment_distance(seg, point; tol=tol))
    end
    distance
end 


"""Distances to more points - which are the columns of the matrix"""
function curve_distance(Γ::Curve, points::Matrix{T}; tol=1E-10) where T<:AbstractFloat
    gdim, npoints = size(points)
    distances = Vector{Float64}(undef, npoints)
    for (i, point) ∈ enumerate(eachcol(points))
        distances[i] = curve_distance(Γ, point; tol=tol)
    end
    distances
end


"""Vector of Vectors to Matrix"""
function vecvec_to_mat(vecvec::AbstractVector{T}) where T 
    ncols = length(vecvec)
    nrows = length(vecvec[1])
    mat = Array{eltype(vecvec[1]), 2}(undef, nrows, ncols)
    @inbounds @fastmath for col in 1:ncols, row in 1:nrows
        mat[row, col] = vecvec[col][row]
    end
    return mat
end


"""Construct SegmentMesh from Boundary of 2d mesh"""
function SegmentMesh(mesh::CompositeTriangulation)

    cells, pv_to_v = compute_embedding(mesh)

    parent_nodes = mesh.dtrian.trian.grid.parent.node_coordinates
    nodes = Vector{eltype(parent_nodes)}(undef, length(pv_to_v))
    for (v, pv) ∈ enumerate(pv_to_v)
        nodes[v] = parent_nodes[pv]
    end
    SegmentMesh(vecvec_to_mat(nodes), cells)
end 


"""How we view entities of mesh from the segment mesh"""
function compute_embedding(mesh::CompositeTriangulation)
    # We have a mesh here as a view of facets of parent
    parent_edges = mesh.dtrian.trian.grid.parent.cell_node_ids
    cell_to_parent_cell = mesh.dtrian.trian.grid.cell_to_parent_cell

    # Cell 2 vertex in the local numbering
    cells = Vector{Vector{Int}}()
    resize!(cells, length(cell_to_parent_cell))
    # Mapping of the vertices
    pv_to_v = Dict{Int, Int}()
    for (cell, parent_cell) ∈ enumerate(cell_to_parent_cell)
        pvs = parent_edges[parent_cell]  # These are parent vertex indices

        vs = Vector{Int}(undef, 2)
        # Encode cell in numbering of the new mesh
        for i ∈ eachindex(pvs)
            pv = pvs[i]
            v = -1
            if pv ∉ keys(pv_to_v)
                v = length(pv_to_v) + 1
                pv_to_v[pv] = v 
                vs[i] = v
            else
                vs[i] = pv_to_v[pv]
            end
        end
        cells[cell] = vs
    end
    # Since pv_to_v is one-to-one
    vertices = Vector{Int}(undef, length(pv_to_v))
    for (pv, v) ∈ pv_to_v
        vertices[v] = pv
    end
    (cells, vertices)
end



# Gridap interoperaobity
Curve(mesh::CompositeTriangulation) = Curve(SegmentMesh(mesh))


"""Distance computation with signature suitable for gridap"""
function curve_distance(Γ::Curve, point::VectorValue{2, T}; tol=1E-10) where T<:Real
    # And dispatch to the above
    curve_distance(Γ, [point[1], point[2]]; tol=tol)
end

# FIXME: curve_distance(Γ, space)

# -------------------

function min_facet_area(mesh::Triangulation)
    Γ = BoundaryTriangulation(mesh)
    dΓ = Measure(Γ, 1)

    γ = SkeletonTriangulation(mesh)
    dγ = Measure(γ, 1)

    hs = min(minimum(get_array(∫(1)*dΓ)),
             minimum(get_array(∫(1)*dγ)))
end


"""Represent line mesh as a weighted graph"""
function GraphMesh(mesh::SegmentMesh)
    nvtx = length(mesh.vertices)
    G = SimpleWeightedGraph(nvtx)
    for (v1, v2) ∈ mesh.topology
        d = norm(mesh.vertices[:, v1] - mesh.vertices[:, v2])
        # @show (v1, v2, d, mesh.vertices[:, v1], mesh.vertices[:, v2])
        @assert d > 0
        @assert add_edge!(G, v1, v2, d)
    end
    G
end


GraphMesh(mesh::CompositeTriangulation) = GraphMesh(SegmentMesh(mesh))


# FIXME: make into a function
false && begin 
    model_path, normals = split_square_mesh(1., offset=0.2, distance=2)
    model = GmshDiscreteModel(model_path)

    writevtk(model, "test_model")

    Ω = Triangulation(model)
    Γ = BoundaryTriangulation(Ω, tags=["interface"])
    dΓ = Measure(Γ, 1)
    reference = sum(∫(1)*dΓ)

    # We want to represent Γ as a graph and compute distances between marked points on it
    G = GraphMesh(Γ)
    # One thing that remains is that the tags for points are defined with respect to parent mesh 
    fb = get_face_labeling(model)
    point_tags = get_face_tag(fb, 0)
    # Lookup in parent numbering (of Ω)
    tag_l = get_tag_from_name(fb, "iface_left")
    pidx_l = findfirst(x -> x == tag_l, point_tags)

    tag_r = get_tag_from_name(fb, "iface_right")
    pidx_r = findfirst(x -> x == tag_r, point_tags)
    # Nowe we want to convert them to local numbering of Γ 
    _, child_to_parent_vertex = compute_embedding(Γ)

    cidx_l = findfirst(x -> x == pidx_l, child_to_parent_vertex)
    cidx_r = findfirst(x -> x == pidx_r, child_to_parent_vertex)
    # We can now query the graph
    path = child_to_parent_vertex[enumerate_paths(dijkstra_shortest_paths(G, cidx_l), cidx_r)]
    nodes = Ω.grid.node_coordinates   
    distance = sum(norm(nodes[path[i]] .- nodes[path[i+1]]) for i ∈ 1:(length(path)-1))
    @show (distance, reference)
end


false && begin
    #= # Simple shape
    vertices = [0 0;
                1 0.;
                0.5 0.5;
                0.5 0.75;
                0.25 0.6]'

    vertices = collect(vertices)
    topology = [[1, 3], 
                [2, 3],
                [3, 4],
                [4, 5]]
    =#

    # Stress test: computing distance from a collection of segments 
    # representing circle
    npts = 128
    θ = collect(range(0, 2π, npts+1))[1:end-1]
    # As distance from circle at 0.5, 0.5 with radius
    ρ = 0.25
    x₀, x₁ = 0.5, 0.5
    vertices = hcat(x₀ .+ ρ*sin.(θ), x₁ .+ ρ*cos.(θ))

    vertices = collect(vertices')
    topology = [[i, 1+i%npts] for i ∈ 1:npts]

    mesh = SegmentMesh(vertices, topology)
    Γ = Curve(mesh)

    # Pick up the curve from marked facets in the mesh; here 
    # we take marking as the boundary
    #=
    model = CartesianDiscreteModel((0, 1, 0, 1), (64, 64))
    Ω = Triangulation(model)
    mesh_ = BoundaryTriangulation(Ω)
    Γ = Curve(mesh_)
    =#

    ncells = 256
    # Now the mesh on which we want to evl the distance field
    model = CartesianDiscreteModel((0, 1, 0, 1), (ncells, ncells))
    Ω = Triangulation(model)
    # Let's see about it in P0 space
    Velm = ReferenceFE(lagrangian, Float64, 0)
    V = TestFESpace(Ω, Velm; conformity=:L2)

    # Some notion of mesh resolution
    distΓ(x) = curve_distance(Γ, x)
    
    dx = min_facet_area(Ω)
    maskΓ(x) = distΓ(x) > dx ? 0 : 1

    f = interpolate_everywhere(distΓ, V)
    mask = interpolate_everywhere(maskΓ, V)

    writevtk(Ω, "dist", order=1, cellfields=["dist" => f,
                                             "mask" => mask])

    true_distΓ(x) = abs(sqrt((x[1]-x₀)^2 + (x[2]-x₁)^2)-ρ)
    dΩ = Measure(Ω, 5)
    ef = f - true_distΓ
    L = ∫(ef*ef)*dΩ
    @show sqrt(sum(L))
end