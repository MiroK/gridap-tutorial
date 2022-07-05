using GridapGmsh: gmsh, GmshDiscreteModel


"""[0, 1]^2 with tri/quad cells
  __3__
4|     |2
 |__1__|
"""
function unit_square_mesh(clscale::Real, cell_type::Symbol=:quad; distance::Real=Inf, structured::Bool=false, view::Bool=false, save::String="unit_square")
    @assert clscale > 0 
    @assert cell_type ∈ (:quad, :tri)
    @assert !isempty(save)
    @assert distance == Inf || distance > 0
    @assert !(distance < Inf && structured)

    !isdir(".msh_cache") && mkdir(".msh_cache")
    save = joinpath(".msh_cache", save)

    gmsh.initialize(["", "-clscale", string(clscale)])

    model = gmsh.model
    occ = model.occ

    make_line = (p, q) -> occ.addLine(p, q)

    points = []
    if distance == Inf
        push!(points, [occ.addPoint(0, 0, 0), 
                       occ.addPoint(1, 0, 0), 
                       occ.addPoint(1, 1, 0), 
                       occ.addPoint(0, 1, 0)]...)
        lines = [make_line(points[p], points[1 + p%4]) for p ∈ eachindex(points)] 
    else
        θ = 1/2/distance
        center = distance*[sin(θ), -cos(θ)]
        ϕ = (π-θ)/2

        center_id = occ.addPoint(center[1], center[2], 0)
        make_arc = (p, q) -> occ.addCircleArc(p, center_id, q)

        A = center + distance*[-cos(ϕ), sin(ϕ)]
        B = center + distance*[cos(ϕ), sin(ϕ)]
        C = center + (1+distance)*[cos(ϕ), sin(ϕ)]
        D = center + (1+distance)*[-cos(ϕ), sin(ϕ)]

        for p ∈ (A, B, C, D)
            push!(points, occ.addPoint(p[1], p[2], 0))
        end

        lines = [!isodd(i) ? make_line(points[p], points[1 + p%4]) : make_arc(points[p], points[1 + p%4]) 
                 for (i, p) ∈ enumerate(eachindex(points))] 
    end

    loop = occ.addCurveLoop(lines)
    surf = occ.addPlaneSurface([loop])
    occ.synchronize()

    names = ("ll", "lr", "ur", "ul")
    for (tag, (point, name)) ∈ enumerate(zip(points, names))
        point_group = model.addPhysicalGroup(0, [point], tag)
        gmsh.model.setPhysicalName(0, point_group, name)
    end

    surf_group = model.addPhysicalGroup(2, [surf], 1)
    gmsh.model.setPhysicalName(2, surf_group, "surface")

    names = ("bottom", "right", "top", "left")
    for (tag, (line, name)) ∈ enumerate(zip(lines, names))
        line_group = model.addPhysicalGroup(1, [line], tag)
        gmsh.model.setPhysicalName(1, line_group, name)
    end

    gmsh.model.occ.synchronize()

    structured && model.mesh.setTransfiniteSurface(surf)

    cell_type == :quad && gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.model.mesh.generate(2)

    if view
        gmsh.fltk.initialize()
        gmsh.fltk.run()
    end

    name = "$(save)_$(string(clscale)).msh"
    gmsh.write(name)

    gmsh.finalize()

    name
end

using Gridap, GridapGmsh

name = unit_square_mesh(1)
model = GmshDiscreteModel(name)
labels = get_face_labeling(model)

χmap = Dict("bottom" => (x; tol=1E-8) -> (abs(x[2]) < tol && -tol < x[1] < 1+tol) ? 1. : 0.,
            "top" => (x; tol=1E-8) -> (abs(x[2]-1) < tol && -tol < x[1] < 1+tol) ? 1. : 0.,
            "left" => (x; tol=1E-8) -> (abs(x[1]) < tol && -tol < x[2] < 1+tol) ? 1. : 0.,
            "right" => (x; tol=1E-8) -> (abs(x[1]-1) < tol && -tol < x[2] < 1+tol) ? 1. : 0.)

for (tag, χ) ∈ χmap
    Γ = BoundaryTriangulation(model, labels, tags=tag)
    dΓ = Measure(Γ, 1)

    e = sum(∫(χ)*dΓ)
    el = sum(∫(1)*dΓ)
    @show (tag, e, el)
    for (tag1, χ) in χmap
        if tag != tag1
            ee = sum(∫(χ)*dΓ)
            @show (" ", tag1, ee)
        end
    end 
end