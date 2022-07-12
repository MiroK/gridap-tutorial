using NearestNeighbors
using LinearAlgebra


struct SegmentMesh{T<:Real}
    vertices::Matrix{T}
    topology::Dict{Int, Vector{Int}}
end


struct Curve{T<:Real}
    tree::KDTree
    vertex2cell::Dict{Int, Vector{Int}}
    # Curve encoding
    vertices::Matrix{T}
    cell2vertex::Dict{Int, Vector{Int}}
end


function Curve(mesh::SegmentMesh; leafsize=10)
    vertices = mesh.vertices
    cell2vertex = mesh.topology
    tree = KDTree(vertices; leafsize=leafsize)

    # Invert the connectivity
    vertex2cell = Dict{Int, Vector{Int}}()
    for (cell, vertices) ∈ cell2vertex
        for vertex ∈ vertices
            if vertex ∉ keys(vertex2cell) 
                vertex2cell[vertex] = Vector{Int}()
            end
            append!(vertex2cell[vertex], cell)
        end
    end

    Curve(tree, vertex2cell, vertices, cell2vertex)
end


function curve_distance(Γ::Curve{T}, point; tol=1E-10) where T<:Real
    (vindex, distance) = nn(Γ.tree, point)

    vertices = Γ.vertices
    vertex2cell, cell2vertex = Γ.vertex2cell, Γ.cell2vertex
    # Now we look up cells/segments that have this get_boundary_tags
    segments = [Segment(vertices[:, cell2vertex[cell][1]], vertices[:, cell2vertex[cell][2]]) for cell ∈ vertex2cell[vindex]]
    distance = Inf
    for seg ∈ segments
        distance = min(distance, segment_distance(seg, point; tol=tol))
    end
    distance
end 


function curve_distance(Γ::Curve{S}, points::Matrix{T}; tol=1E-10) where S<:Real where T<:Real
    gdim, npoints = size(points)
    distances = Vector{Real}(undef, npoints)
    for (i, point) ∈ enumerate(eachcol(points))
        distances[i] = curve_distance(Γ, point; tol=tol)
    end
    distances
end

struct Segment{T<:Real, S<:Real}
    A::Vector{T}
    B::Vector{S}
end


function segment_distance(segment::Segment, point; tol=1E-10)
    A, B = segment.B, segment.A
    ν = B .- A
    l = LinearAlgebra.norm(ν)^2
    s = dot(point .- A, ν)/l

    d = min(norm(point .- A), norm(point .- B))
    (s > 1 + tol || s < -tol) && return d

    X = A .+ ν*s
    min(d, norm(point .- X))
end

# -------------------

vertices = [0 0;
            2 0.;
            1 1.;
            1 2.;
            2 2.]'

vertices = collect(vertices)
topology = Dict(1 => [1, 3], 
                2 => [2, 3],
                3 => [3, 4],
                4 => [4, 5])

mesh = SegmentMesh(vertices, topology)
curve = Curve(mesh)

curve_distance(curve, [0.4, 0.5])
@show curve_distance(curve, rand(2, 4))
