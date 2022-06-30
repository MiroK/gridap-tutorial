using Gridap
using Printf

include("GridapUtils.jl")
using .GridapUtils
using Symbolics 


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

    # Define Dirichlet boundaries
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "dirichlet", Dirichlet_tags)

    writevtk(model,"model")

    reffe = ReferenceFE(lagrangian, Float64, pdegree)
    V0 = TestFESpace(model, reffe; conformity=:H1, dirichlet_tags="dirichlet")

    Ug = TrialFESpace(V0, g0)

    degree = pdegree+1
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 2*degree)

    # Bilinear form
    a(u,v) = ∫( ∇(v)⋅∇(u) )*dΩ

    Γ = BoundaryTriangulation(model, tags=Neumann_tags)
    ν = get_normal_vector(Γ)
    dΓ = Measure(Γ, degree)
    b(v) = ∫( v*f0 )*dΩ + ∫(v*(h0⋅ν))*dΓ

    one(x) = 1
    !isempty(Neumann_tags) && @assert sqrt(sum( ∫(one)*dΓ )) > 0
    isempty(Neumann_tags) && @assert sqrt(sum( ∫(one)*dΓ )) < 1E-10 
    op = AffineFEOperator(a, b, Ug, V0)

    ls = LUSolver()
    solver = LinearFESolver(ls)

    uh = solve(solver, op)

    return uh
end

true && begin

    x = Symbolics.variables(:x, 1:2)

    # Manufactured solution
    # u0_ = cos(π*(x[1]*x[2]))
    # u0_ = cos(2*π*x[1])*cos(3*π*x[2]) + 1
    u0_ = cos(2*π*(x[1]-x[2]))

    flux_ = -Grad(u0_)
    f0_ = Div(flux_)
    h0_ = Grad(u0_)

    u0, f0, h0 = (compile(arg, x) for arg in (u0_, f0_, h0_))

    Dirichlet_tags = [5, 6]

    pdegree = 1 # Polynomial degree of FE space

    global uh
    errors, hs, ndofs = [], [], []
    for k ∈ 2:6
        n = 2^(k+2)

        domain = (0,1,0,1)
        partition = (n, n)
        model = CartesianDiscreteModel(domain, partition)
        model = simplexify(model)

        global uh = poisson_solver(model, f0, u0, h0, Dirichlet_tags; pdegree=pdegree)

        Ω = get_triangulation(uh)
        dΩ = Measure(Ω, 2*pdegree)

        e = u0 - uh
        error = sqrt(sum( ∫( e*e + ∇(e)⋅∇(e) )*dΩ ))
        append!(errors, error)
        append!(hs, minimum(get_mesh_sizes(Ω)))
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