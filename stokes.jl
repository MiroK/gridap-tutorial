using Gridap
using Printf

include("GridapUtils.jl")
using .GridapUtils
using Symbolics 


"""
Solve on Ω the Stokes problem with symmetric gradient 
"""
function stokes_solver(model, f0, g0, h0, Dirichlet_tags; pdegree=2)
    @assert pdegree > 1
    @assert !isempty(Dirichlet_tags)
    # Allowed tags are ...
    all_tags = GridapUtils.get_boundary_tags(model)
    @assert all(t ∈ all_tags for t ∈ Dirichlet_tags)
    Neumann_tags = filter(x -> x ∉ Dirichlet_tags, all_tags)

    # Define Dirichlet boundaries
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "dirichlet", Dirichlet_tags)

    writevtk(model,"model")

    Velm = ReferenceFE(lagrangian, VectorValue{2, Float64}, pdegree)
    V = TestFESpace(model, Velm; conformity=:H1, dirichlet_tags="dirichlet")

    Qelm = ReferenceFE(lagrangian, Float64, pdegree-1)
    
    has_zeromean = isempty(Neumann_tags)
    Q = TestFESpace(model, Qelm; conformity=:H1)
    if has_zeromean
        Q = TestFESpace(model, Qelm; conformity=:H1, constraint=:zeromean)
    end

    U = TrialFESpace(V, g0)
    P = Q

    dW = MultiFieldFESpace([V, Q])
    W = MultiFieldFESpace([U, P])

    degree = pdegree+1
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 2*degree)

    ϵ(v) = (∇(v) + transpose(∇(v)))/2
    # Bilinear form
    a((u, p), (v, q))  = ∫( ϵ(v) ⊙ ϵ(u) )*dΩ - ∫(p*(∇⋅v))*dΩ - ∫(q*(∇⋅u))*dΩ

    Γ = BoundaryTriangulation(model, tags=Neumann_tags)
    ν = get_normal_vector(Γ)
    dΓ = Measure(Γ, 2*degree)
    b((v, q)) = ∫( v⋅f0 )*dΩ + ∫(v⋅(h0⋅ν))*dΓ

    one(x) = 1
    !isempty(Neumann_tags) && @assert sqrt(sum( ∫(one)*dΓ )) > 0
    isempty(Neumann_tags) && @assert sqrt(sum( ∫(one)*dΓ )) < 1E-10 
    op = AffineFEOperator(a, b, W, dW)

    ls = LUSolver()
    solver = LinearFESolver(ls)

    uh = solve(solver, op)

    return uh, has_zeromean
end

true && begin

    x = Symbolics.variables(:x, 1:2)

    # Manufactured solution
    ϕ0_ = sin(π*x[1]) + sin(2*π*x[2])
    u0_ = Rot(ϕ0_)
    p0_ = sin(π*x[1]) - cos(2*π*x[2])
    σ0_ = Sym(Grad(u0_)) - [p0_ 0; 0 p0_]
    f0_ = -Div(σ0_)

    uu0, pp0, ff0, hh0 = (compile(arg, x) for arg in (u0_, p0_, f0_, σ0_))

    Dirichlet_tags = [5, 6, 7]

    pdegree = 2 # Polynomial degree of FE space

    global wh
    errors_u, errors_p, ndofs_u, ndofs_p, hs = [], [], [], [], []
    for k ∈ 2:6
        n = 2^(k)

        domain = (0, 1, 0, 1)
        partition = (n, n)
        model = CartesianDiscreteModel(domain, partition)
        model = simplexify(model)

        global wh, has_zeromean = stokes_solver(model, ff0, uu0, hh0, Dirichlet_tags; pdegree=pdegree)

        uh, ph = wh
        Ω = get_triangulation(uh)
        dΩ = Measure(Ω, 2*pdegree)

        uh, ph = wh
        # Velocity error
        e = uu0 - uh
        error = sqrt(sum( ∫( e⋅e + ∇(e)⊙∇(e) )*dΩ ))
        append!(errors_u, error)
        append!(ndofs_u, length(get_free_dof_ids(uh.fe_space)))

        # What is the mean of true
        volume = sum( ∫(1)*dΩ )
        mean_pressure0 = sum( ∫(pp0)*dΩ )
        mean_pressureh = sum( ∫(ph)*dΩ )

        e = pp0 - ph
        if has_zeromean
            e = pp0 - (ph - mean_pressureh/volume+mean_pressure0/volume)
        end 
        error = sqrt(sum( ∫( e⋅e )*dΩ ))
        append!(errors_p, error)
        append!(ndofs_p, length(get_free_dof_ids(ph.fe_space)))

        append!(hs, minimum(GridapUtils.get_mesh_sizes(Ω)))
    end
    uh, ph = wh
    writevtk(get_triangulation(uh), "stokes_sol", order=1, cellfields=["uh" => uh, "ph" => ph])

    rates_u = log.(errors_u[2:end]./errors_u[1:end-1])./log.(hs[2:end]./hs[1:end-1])
    rates_u = [NaN; rates_u]

    rates_p = log.(errors_p[2:end]./errors_p[1:end-1])./log.(hs[2:end]./hs[1:end-1])
    rates_p = [NaN; rates_p]
    table = hcat(hs, ndofs_u, errors_u, rates_u, ndofs_p, errors_p, rates_p)
    for row in eachrow(table)
        @printf "h = %.2E dim(V) = %d |u-uh|_1 = %.4E rate = %.2f dim(Q) = %d |p-ph|_0 = %.4E rate = %.2f\n" row...
    end
end