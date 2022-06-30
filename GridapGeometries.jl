using GridapGmsh: gmsh, GmshDiscreteModel


"""[0, 1]^2 with tri/quad cells
  __3__
4|     |2
 |__1__|
"""
function unit_square_mesh(clmax::Real, cell_type::Symbol=:quad; structured::Bool=false, view::Bool=false, save::String="unit_square")
    @assert clmax > 0 
    @assert cell_type ∈ (:quad, :tri)
    @assert !isempty(save)
    
    gmsh.initialize(["", "-clmax", string(clmax)])

    model = gmsh.model
    occ = model.occ

    points = [occ.addPoint(0, 0, 0), 
              occ.addPoint(1, 0, 0), 
              occ.addPoint(1, 1, 0), 
              occ.addPoint(0, 1, 0)]

    lines = [occ.addLine(points[p], points[1 + p%4]) for p ∈ eachindex(points)] 
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

    name = "$(save)_$(string(clmax)).msh"
    gmsh.write(name)

    gmsh.finalize()

    name
end


"""Pretty much this
____________
|__STOKES__|
|__DARCY___|
"""
function split_square_mesh(clmax::Real, cell_type::Symbol=:quad; structured::Bool=false, view::Bool=false, save::String="split_square")
    @assert clmax > 0 
    @assert cell_type ∈ (:quad, :tri)
    @assert !isempty(save)
    
    gmsh.initialize(["", "-clmax", string(clmax)])

    model = gmsh.model
    occ = model.occ

    points = [occ.addPoint(0, 0, 0),          
              occ.addPoint(0, 0.5, 0), 
              occ.addPoint(1, 0.5, 0), 
              occ.addPoint(1, 0, 0),
              occ.addPoint(1, -0.5, 0),
              occ.addPoint(0, -0.5, 0)]

    npts = length(points)
    lines = [occ.addLine(points[p], points[1 + p%npts]) for p ∈ eachindex(points)] 
    # Add the interface
    append!(lines, occ.addLine(points[1], points[4]))

    top_lines = [lines[1], lines[2], lines[3], -lines[end]]
    top_loop = occ.addCurveLoop(top_lines)
    top_surf = occ.addPlaneSurface([top_loop])

    bottom_lines = [lines[4], lines[5], lines[6], lines[end]]
    bottom_loop = occ.addCurveLoop(bottom_lines)
    bottom_surf = occ.addPlaneSurface([bottom_loop])

    occ.synchronize()

    # Mark the boundary points of the interface top_loop
    iface_left = model.addPhysicalGroup(0, [points[1]], 1)
    gmsh.model.setPhysicalName(0, iface_left, "iface_left")

    iface_right = model.addPhysicalGroup(0, [points[4]], 2)
    gmsh.model.setPhysicalName(0, iface_right, "iface_right")

    top_surf_group = model.addPhysicalGroup(2, [top_surf], 1)
    gmsh.model.setPhysicalName(2, top_surf_group, "top_surface")

    bottom_surf_group = model.addPhysicalGroup(2, [bottom_surf], 2)
    gmsh.model.setPhysicalName(2, bottom_surf_group, "top_surface")

    names = ("top_left", "top", "top_right", "bottom_right", "bottom", "bottom_left", "interface")
    for (tag, (line, name)) ∈ enumerate(zip(lines, names))
        line_group = model.addPhysicalGroup(1, [line], tag)
        gmsh.model.setPhysicalName(1, line_group, name)
    end

    gmsh.model.occ.synchronize()

    structured && model.mesh.setTransfiniteSurface(top_surf)
    structured && model.mesh.setTransfiniteSurface(bottom_surf)

    cell_type == :quad && gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.model.mesh.generate(2)

    if view
        gmsh.fltk.initialize()
        gmsh.fltk.run()
    end

    name = "$(save)_$(string(clmax)).msh"
    gmsh.write(name)

    gmsh.finalize()

    name
end