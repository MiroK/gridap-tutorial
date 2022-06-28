
using Gridap
using Gridap.Geometry
using Printf


"""
Assuming 2d model get tags leading to nonzero surface integrals
"""
function get_boundary_tags(model)
    labels = get_face_labeling(model)
    # FIXME: can we figure out the degree here from the model?

    Γ = BoundaryTriangulation(model)
    dΓ = Measure(Γ, 0)
    target = sum(∫(1)*dΓ)

    maybe = unique(get_face_tag(labels, 1))
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
Solve on Ω the Poisson -Δ u = f0 with u = g0 on 
Dirichlet part of the boundary and Neumann bcs h0 on the rest 
"""
function poisson_solver(model, f0, g0, h0, Dirichlet_tags; pdegree)
    @assert !isempty(Dirichlet_tags)
    # Allowed tags are ...
    all_tags = get_boundary_tags(model)
    @assert all(t ∈ all_tags for t ∈ Dirichlet_tags)
    Neumann_tags = filter(x -> x ∉ Dirichlet_tags, all_tags)

    # FIXME: which are the tags that we actually have?

    # Define Dirichlet boundaries
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "dirichlet", Dirichlet_tags)

    writevtk(model,"model")

    reffe = ReferenceFE(lagrangian, Float64, pdegree)
    V0 = TestFESpace(model, reffe; conformity=:H1, dirichlet_tags="dirichlet")

    Ug = TrialFESpace(V0, g0)

    degree = pdegree+1
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)

    # Bilinear form
    a(u,v) = ∫( ∇(v)⋅∇(u) )*dΩ

    Γ = BoundaryTriangulation(model, tags=Neumann_tags)
    dΓ = Measure(Γ, degree)
    b(v) = ∫( v*f0 )*dΩ + ∫( v*h0 )*dΓ

    one(x) = 1
    !isempty(Neumann_tags) && @assert sqrt(sum( ∫(one)*dΓ )) > 0

    op = AffineFEOperator(a, b, Ug, V0)

    ls = LUSolver()
    solver = LinearFESolver(ls)

    uh = solve(solver, op)

    return uh
end

true && begin
    # Manufactured solution
    u0(x;k=2, l=1) = cos(k*π*x[1])*cos(l*π*x[2]) + 1
    # Boundary data comes from the true solution
    g0 = u0
    f0(x;k=2, l=1) = ((k*π)^2 + (l*π)^2)*cos(k*π*x[1])*cos(l*π*x[2])
    h0(x) = 0

    pdegree = 3  # Polynomial degree of FE space

    global uh
    errors, hs, ndofs = [], [], []
    for k ∈ 2:6
        n = 2^k

        domain = (0,1,0,1)
        partition = (n, n)
        model = CartesianDiscreteModel(domain, partition)

        Dirichlet_tags = [5, 6, 7]
        global uh = poisson_solver(model, f0, g0, h0, Dirichlet_tags; pdegree=pdegree)

        Ω = get_triangulation(uh)
        dΩ = Measure(Ω, 4)

        e = u0 - uh
        error = sqrt(sum( ∫( e*e + ∇(e)⋅∇(e) )*dΩ ))
        append!(errors, error)
        append!(hs, 1/n)
        append!(ndofs, length(get_free_dof_ids(uh.fe_space)))
    end

    writevtk(get_triangulation(uh), "poisson_sol", order=1, cellfields=["uh" => uh])

    rates = log.(errors[2:end]./errors[1:end-1])./log.(hs[2:end]./hs[1:end-1])
    rates = [NaN; rates]
    table = hcat(hs, ndofs, errors, rates)
    for row in eachrow(table)
        @printf "h = %.2E dim(V) = %d |u-uh|_1 = %.4E rate = %.2f\n" row...
    end
end
