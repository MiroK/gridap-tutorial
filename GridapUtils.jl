module GridapUtils

using Gridap
using Gridap.Geometry

"""
Get get tags leading to nonzero surface integrals
"""
function get_boundary_tags(model)
    labels = get_face_labeling(model)
    # FIXME: can we figure out the degree here from the model?

    Γ = BoundaryTriangulation(model)
    dΓ = Measure(Γ, 0)
    target = sum(∫(1)*dΓ)

    tdim = num_cell_dims(Γ)
    maybe = unique(get_face_tag(labels, tdim))
    lengths = Dict{eltype(maybe), Float64}()
    for tag ∈ maybe
        Γ = BoundaryTriangulation(model, tags=[tag])
        dΓ = Measure(Γ, 0)
        l = sum(∫(1)*dΓ)

        l < target && setindex!(lengths, l, tag)
    end

    @assert isapprox(sum(values(lengths)), target; rtol=1E-8)

    return collect(keys(lengths))
end

"""
Approximate min and max of the cell size computed from volume
"""
function get_mesh_sizes(mesh::Triangulation)
    dΩ = Measure(mesh, 0)
    vols = get_array(∫(1)dΩ)
    tdim = num_cell_dims(mesh)
    return extrema(vols.^(1/tdim))
end

export get_boundary_tags, get_mesh_sizes

# ----------------------------------------------------------------------------

include("GridapGeometries.jl")

export unit_square_mesh, split_square_mesh

# -----------------------------------------------------------------------------

include("GridapSym.jl")

export Grad, Div, Curl, Rot, Dot, Inner, Sym, Skew
export compile

end