using LinearAlgebra
using Gridap.Geometry: CompositeTriangulation


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
    # We have a mesh here as a view of facets of parent
    parent_edges = mesh.dtrian.trian.grid.parent.cell_node_ids
    parent_nodes = mesh.dtrian.trian.grid.parent.node_coordinates
    cell_to_parent_cell = mesh.dtrian.trian.grid.cell_to_parent_cell

    nodes = Vector{eltype(parent_nodes)}()
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
                push!(nodes, parent_nodes[pv])
                v = length(nodes)
                pv_to_v[pv] = v 
                vs[i] = v
            else
                vs[i] = pv_to_v[pv]
            end
        end
        cells[cell] = vs
    end

    SegmentMesh(vecvec_to_mat(nodes), cells)
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

vertices = [0 0;
            1 0.;
            1 1.;
            0 1.]'

vertices = collect(vertices)
topology = [[1, 2], 
            [2, 3],
            [3, 4],
            [4, 1],
            [1, 3]]

mesh = SegmentMesh(vertices, topology)
Γ = Curve(mesh)

curve_distance(Γ, [0.4, 0.5])
#curve_distance(curve, rand(2, 4))

# Define the curve from some edges
model = CartesianDiscreteModel((0, 1, 0, 1), (64, 64))
Ω = Triangulation(model)
mesh_ = BoundaryTriangulation(Ω)
Γ = Curve(mesh_)

# Now the mesh on which we want to evl the distance field
model = CartesianDiscreteModel((0, 1, 0, 1), (100, 100))
Ω = Triangulation(model)
# Let's see about it in P1 space
Velm = ReferenceFE(lagrangian, Float64, 1)
V = TestFESpace(Ω, Velm; conformity=:H1)

distΓ(x) = curve_distance(Γ, x)

f = interpolate_everywhere(distΓ, V)
writevtk(Ω, "dist", order=1, cellfields=["dist" => f])