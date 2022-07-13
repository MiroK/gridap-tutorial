using LinearAlgebra
using Gridap.Geometry: CompositeTriangulation
using Gridap


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

function min_facet_area(mesh::Triangulation)
    Γ = BoundaryTriangulation(mesh)
    dΓ = Measure(Γ, 1)

    γ = SkeletonTriangulation(mesh)
    dγ = Measure(γ, 1)

    hs = min(minimum(get_array(∫(1)*dΓ)),
             minimum(get_array(∫(1)*dγ)))
end

begin
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
    Γ = Curve(mesh_)=#

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