using GridapGmsh: gmsh, GmshDiscreteModel
using Symbolics: variables, Num
using LinearAlgebra


"""Normal vector of segment"""
function seg_normal(A, B)
    τ = B - A
    n = [0 -1; 1 0]*τ
    normal = n/norm(n)
    Vector{Num}(normal)
end


"""Normal vector of circular arc"""
function arc_normal(A, C, B)
    r = norm(A-C)
    x = variables(:x, 1:2)

    n = [(x[1]-C[1])/r, (x[2]-C[2])/r] 
end

function cross2d(u::Vector{T}, v::Vector{S}) where S <: Real where T<: Real
    @assert length(u) == length(v) == 2
    u[2]*v[1] - u[1]*v[2]
end

"""
Mesh for geometry in 2D marked by points
"""
function polygon_mesh(points::Matrix{T}, clscale::Real, cell_type::Symbol=:quad; view::Bool=false, save::String="polygon") where T<:Real
    npoints, gdim = size(points)
    @assert gdim == 2 && npoints >= 3
    @assert !isapprox(norm(points[1]-points[end]), 0)
    @assert clscale > 0 
    @assert cell_type ∈ (:quad, :tri)
    @assert !isempty(save)

    A, B, C = points[1, :], points[2, :], points[3, :]
    orientation = sign(cross2d(B-A, C-A))
    for i ∈ 2:(npoints-2)
        A, B, C = points[i, :], points[i+1, :], points[i+2, :]
        @assert orientation == sign(cross2d(B-A, C-A))
    end

    !isdir(".msh_cache") && mkdir(".msh_cache")
    save = joinpath(".msh_cache", save)

    gmsh.initialize(["", "-clscale", string(clscale)])

    model = gmsh.model
    occ = model.occ

    points = [occ.addPoint(points[i, 1], points[i, 2], 0) for i ∈ 1:npoints]
    lines = [occ.addLine(points[i], points[1 + i%npoints]) for i ∈ eachindex(points)]

    loop = occ.addCurveLoop(lines)
    surf = occ.addPlaneSurface([loop])
    occ.synchronize()

    for (tag, point) ∈ enumerate(points)
        point_group = model.addPhysicalGroup(0, [point], tag)
        name = "v"*string(tag)
        gmsh.model.setPhysicalName(0, point_group, name)
    end

    surf_group = model.addPhysicalGroup(2, [surf], 1)
    gmsh.model.setPhysicalName(2, surf_group, "surface")

    normals = Dict{String, Vector{Num}}()
    for (tag, line) ∈ enumerate(lines)
        line_group = model.addPhysicalGroup(1, [line], tag)

        up, (v2id, v1id) = gmsh.model.getAdjacencies(1, tag)
        
        name = "l"*string(v1id)*"_"*string(v2id)
        gmsh.model.setPhysicalName(1, line_group, name)

        v1 = gmsh.model.getValue(0, v1id, [])
        v2 = gmsh.model.getValue(0, v2id, [])
        
        n = seg_normal(v1[1:end-1], v2[1:end-1])
        normals[name] = n
    end

    gmsh.model.occ.synchronize()

    cell_type == :quad && gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.model.mesh.generate(2)

    if view
        gmsh.fltk.initialize()
        gmsh.fltk.run()
    end

    name = "$(save)_$(string(clscale)).msh"
    gmsh.write(name)

    gmsh.finalize()

    (name, normals)
end


"""Mesh a circle of radius r and center (0, 0)"""
function circle_mesh(clscale::Real, cell_type::Symbol=:quad; radius::Real=1, view::Bool=false, save::String="circle")
    @assert clscale > 0 
    @assert cell_type ∈ (:quad, :tri)
    @assert !isempty(save)
    @assert 0 < radius

    !isdir(".msh_cache") && mkdir(".msh_cache")
    save = joinpath(".msh_cache", save)

    gmsh.initialize(["", "-clscale", string(clscale)])

    model = gmsh.model
    occ = model.occ

    #   A 
    #D  O  B
    #   C
    O = occ.addPoint(0, 0, 0)
    A = occ.addPoint(0, radius, 0)
    B = occ.addPoint(radius, 0, 0)
    C = occ.addPoint(0, -radius, 0)
    D = occ.addPoint(-radius, 0, 0)
    
    lines = []
    for (P, Q) ∈ ((A, B), (B, C), (C, D), (D, A))
        push!(lines, occ.addCircleArc(P, O, Q))
    end

    loop = occ.addCurveLoop(lines)
    surf = occ.addPlaneSurface([loop])
    occ.synchronize()

    names = ("n", "e", "s", "w")
    for (tag, (point, name)) ∈ enumerate(zip((A, B, C, D), names))
        point_group = model.addPhysicalGroup(0, [point], tag)
        gmsh.model.setPhysicalName(0, point_group, name)
    end

    surf_group = model.addPhysicalGroup(2, [surf], 1)
    gmsh.model.setPhysicalName(2, surf_group, "surface")

    names = ("ne", "se", "sw", "nw")
    for (tag, (line, name)) ∈ enumerate(zip(lines, names))
        line_group = model.addPhysicalGroup(1, [line], tag)
        gmsh.model.setPhysicalName(1, line_group, name)
    end

    normal = arc_normal([0., radius], [0., 0.], [radius, 0.])
    normals = Dict("ne" => normal,
                   "sw" => normal,
                   "sw" => normal,
                   "nw" => normal)

    gmsh.model.occ.synchronize()

    cell_type == :quad && gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.model.mesh.generate(2)

    if view
        gmsh.fltk.initialize()
        gmsh.fltk.run()
    end

    name = "$(save)_$(string(clscale)).msh"
    gmsh.write(name)

    gmsh.finalize()

    (name, normals)
end


"""Disk  tri/quad cells
  __3__
4|     |2
 |__1__|
"""
function disk_mesh(clscale::Real, cell_type::Symbol=:quad; radius0::Real=0.5, radius1::Real=1, angle=π/4, view::Bool=false, save::String="disk")
    @assert clscale > 0 
    @assert cell_type ∈ (:quad, :tri)
    @assert !isempty(save)
    @assert 0 < radius0 < radius1
    @assert 0 < angle < π

    !isdir(".msh_cache") && mkdir(".msh_cache")
    save = joinpath(".msh_cache", save)

    gmsh.initialize(["", "-clscale", string(clscale)])

    model = gmsh.model
    occ = model.occ

    ϕ = (π-angle)/2

    O = [0, 0.]
    A = radius0*[-cos(ϕ), sin(ϕ)]
    B = radius0*[cos(ϕ), sin(ϕ)]
    C = radius1*[cos(ϕ), sin(ϕ)]
    D = radius1*[-cos(ϕ), sin(ϕ)]

    center = occ.addPoint(O[1], O[2], 0)
    points = []
    for p ∈ (A, B, C, D)
        push!(points, occ.addPoint(p[1], p[2], 0))
    end

    lines = [occ.addCircleArc(points[1], center, points[2]),
             occ.addLine(points[2], points[3]),
             occ.addCircleArc(points[3], center, points[4]),
             occ.addLine(points[4], points[1])] 

    x = variables(:x, 1:2)
    normals = Dict("bottom" => -arc_normal(A, O, B),
                   "right" => seg_normal(C, B),
                   "top"  => arc_normal(D, O, C),
                   "left" => seg_normal(A, D))

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

    cell_type == :quad && gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.model.mesh.generate(2)

    if view
        gmsh.fltk.initialize()
        gmsh.fltk.run()
    end

    name = "$(save)_$(string(clscale)).msh"
    gmsh.write(name)

    gmsh.finalize()

    (name, normals)
end


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

    points, normals = [], Dict{String, Vector{Num}}()
    if distance == Inf
        push!(points, [occ.addPoint(0, 0, 0), 
                       occ.addPoint(1, 0, 0), 
                       occ.addPoint(1, 1, 0), 
                       occ.addPoint(0, 1, 0)]...)
        lines = [make_line(points[p], points[1 + p%4]) for p ∈ eachindex(points)] 
    
        # Outward normal vectors
        normals["bottom"] = Vector{Num}([0, -1])
        normals["right"] = Vector{Num}([1, 0])
        normals["top" ] = Vector{Num}([0, 1])
        normals["left"] = Vector{Num}([-1, 0])
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

        x = variables(:x, 1:2)
        normals["bottom"] = -arc_normal(A, center, B)
        normals["right"] = seg_normal(C, B)
        normals["top" ] = arc_normal(D, center, C)
        normals["left"] = seg_normal(A, D)
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

    (name, normals)
end


"""Pretty much this
____________
|__STOKES__|
|__DARCY___|
"""
function split_square_mesh(clscale::Real, cell_type::Symbol=:quad; distance::Real=Inf, offset::Real=0, structured::Bool=false, view::Bool=false, save::String="split_square")
    @assert clscale > 0 
    @assert cell_type ∈ (:quad, :tri)
    @assert !isempty(save)
    @assert distance == Inf || distance > 0
    @assert !(distance < Inf && structured)
    @assert -0.5 < offset < 0.5
    @assert !(abs(offset) > 0 && structured)

    !isdir(".msh_cache") && mkdir(".msh_cache")
    save = joinpath(".msh_cache", save)
    
    gmsh.initialize(["", "-clscale", string(clscale)])

    model = gmsh.model
    occ = model.occ

    make_line = (p, q) -> occ.addLine(p, q)

    points, normals = [], Dict{String, Vector{Num}}()
    if distance == Inf
        push!(points, [occ.addPoint(0, 0, 0),          
                       occ.addPoint(0, 0.5, 0), 
                       occ.addPoint(1, 0.5, 0), 
                       occ.addPoint(1, offset, 0),
                       occ.addPoint(1, -0.5, 0),
                       occ.addPoint(0, -0.5, 0)]...)

        npts = length(points)
        lines = [occ.addLine(points[p], points[1 + p%npts]) for p ∈ eachindex(points)] 
        # Add the interface
        append!(lines, occ.addLine(points[1], points[4]))

        normals["top_left"] = Vector{Num}([-1, 0])
        normals["top"] = Vector{Num}([0, 1])
        normals["top_right"] = Vector{Num}([1, 0])
        normals["bottom_right"] = Vector{Num}([1, 0])
        normals["bottom"] = Vector{Num}([0, -1])
        normals["bottom_left"] = Vector{Num}([-1, 0])
        # (dx, dy) = (1, offset), from bottom to top
        normals["interface"] = Vector{Num}([-offset, 1]/sqrt(1+offset^2))
    else
        θ = 1/2/distance
        center = distance*[sin(θ), -cos(θ)]
        ϕ = (π-θ)/2

        center_id = occ.addPoint(center[1], center[2], 0)

        #  B    C 
        #  A    D
        #  F    E
        pointsX = [center + (distance+0.5)*[-cos(ϕ), sin(ϕ)],
                   center + (distance+1)*[-cos(ϕ), sin(ϕ)],
                   center + (distance+1)*[cos(ϕ), sin(ϕ)],
                   center + (distance+0.5+offset)*[cos(ϕ), sin(ϕ)],
                   center + distance*[cos(ϕ), sin(ϕ)],
                   center + distance*[-cos(ϕ), sin(ϕ)]]
        
        for p ∈ pointsX
            push!(points, occ.addPoint(p[1], p[2], 0))
        end

        npts = length(points)
        # 1, 3, 4, 6
        segments = (1, 3, 4, 6)
        lines = [i ∈ segments ? occ.addLine(points[p], points[1 + p%npts]) : occ.addCircleArc(points[p], center_id, points[1 + p%npts])
                 for (i, p) ∈ enumerate(eachindex(points))] 

        A = center + (distance+0.5)*[-cos(ϕ), sin(ϕ)] 
        B = center + (distance+0.5+offset)*[cos(ϕ), sin(ϕ)]   
        τ = [0 1; -1 0]*(B-A)
        P = (A+B)/2
        @assert abs(dot(τ, B-A)) < 1E-13
        a, b, c = norm(τ, 2)^2, -2*dot(τ, A-P), norm(A-P, 2)^2-distance^2
        s = (-b + sqrt(b^2 - 4*a*c))/2/a
        C = P + s*τ
        @assert abs(norm(A-C)-norm(B-C)) < 1E-13
        @assert abs(norm(A-C)-distance) < 1E-13

        iface_center_id = occ.addPoint(C[1], C[2], 0)

        # Add the interface
        append!(lines, occ.addCircleArc(points[1], iface_center_id, points[4]))

        normals["top_left"] = seg_normal(pointsX[1], pointsX[2])
        normals["top"] = arc_normal(pointsX[2], center, pointsX[3])
        normals["top_right"] = seg_normal(pointsX[3], pointsX[4])
        normals["bottom_right"] = seg_normal(pointsX[4], pointsX[5])
        # Below
        normals["bottom"] = -arc_normal(pointsX[5], center, pointsX[6])
        normals["bottom_left"] = seg_normal(pointsX[6], pointsX[1])
        # (dx, dy) = (1, offset), from bottom to top
        normals["interface"] = arc_normal(pointsX[1], C, pointsX[4])
    end

    top_lines = [lines[1], lines[2], lines[3], -lines[end]]
    top_loop = occ.addCurveLoop(top_lines)
    top_surf = occ.addPlaneSurface([top_loop])

    bottom_lines = [lines[4], lines[5], lines[6], lines[end]]
    bottom_loop = occ.addCurveLoop(bottom_lines)
    bottom_surf = occ.addPlaneSurface([bottom_loop])

    occ.synchronize()

    # Mark the boundary points of the interface top_loop
    names = ("iface_left", "ul", "ur", "iface_right", "lr", "ll")
    for (tag, (point, name)) ∈ enumerate(zip(points, names))
        iface_tag = model.addPhysicalGroup(0, [point], tag)
        gmsh.model.setPhysicalName(0, iface_tag, name)
    end

    top_surf_group = model.addPhysicalGroup(2, [top_surf], 1)
    gmsh.model.setPhysicalName(2, top_surf_group, "top_surface")

    bottom_surf_group = model.addPhysicalGroup(2, [bottom_surf], 2)
    gmsh.model.setPhysicalName(2, bottom_surf_group, "bottom_surface")

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

    name = "$(save)_$(string(clscale)).msh"
    gmsh.write(name)

    gmsh.finalize()

    (name, normals)
end