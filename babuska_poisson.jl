using Gridap
using Gridap.Geometry
using Printf
import Gridap: ∇

"""
Solve on Ω the Poisson -Δ u = f0 with u = g0 on 
Dirichlet part of the boundary and Neumann bcs h0 on the rest.
Problem is formulated using Lagrange multipliers to enforce the 
Dirichlet boundary conditions.
"""
function poisson_solver(model, f0, g0, h0, Dirichlet_tags; pdegree, qdegree)
    @assert !isempty(Dirichlet_tags)
    # Allowed tags are ...
    all_tags = [5, 6, 7, 8]
    @assert all(t ∈ all_tags for t ∈ Dirichlet_tags)
    Neumann_tags = filter(x -> x ∉ Dirichlet_tags, all_tags)

    # FIXME: which are the tags that we actually have?

    # Define Dirichlet boundaries
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "dirichlet", Dirichlet_tags)

    Ω = Triangulation(model)
    ΓD = BoundaryTriangulation(model, tags=Dirichlet_tags)
    ΓN = BoundaryTriangulation(model, tags=Neumann_tags)

    Velm = ReferenceFE(lagrangian, Float64, pdegree)
    Qelm = ReferenceFE(lagrangian, Float64, qdegree)

    V = TestFESpace(Ω, Velm; conformity=:H1)
    Q = TestFESpace(ΓD, Qelm; conformity=:H1) 
    W = MultiFieldFESpace([V, Q])

    dΩ = Measure(Ω, 2*(pdegree-1))
    dΓD = Measure(ΓD, pdegree+qdegree)
    dΓN = Measure(ΓN, pdegree+qdegree)
    # Bilinear form
    a((u, p), (v, q)) = ∫( ∇(v)⋅∇(u) )*dΩ + ∫(v*p)*dΓD + ∫(u*q)*dΓD

    b((v, q)) = ∫( v*f0 )*dΩ + ∫( q*g0 )*dΓD + ∫(v*h0)*dΓN

    op = AffineFEOperator(a, b, W, W)
    return op
end

begin
    # Manufactured solution
    u0(x;k=2, l=1) = cos(k*π*x[1])*cos(l*π*x[2]) + 1
    # NOTE: This function is such that grad(u0) is 0 and in turn 
    # we get very simple case for checking convergence of the multiplier

    g0 = u0 # Boundary data comes from the true solution
    f0(x;k=2, l=1) = ((k*π)^2 + (l*π)^2)*cos(k*π*x[1])*cos(l*π*x[2])
    h0(x) = 0

    pdegree = 2  # Polynomial degree of FE space
    qdegree = pdegree

    whs = []
    errors_u, errors_p, hs, ndofsV, ndofsQ = [], [], [], [], []
    for k ∈ 2:6
        n = 2^k

        domain = (0,1,0,1)
        partition = (n, n)
        model = CartesianDiscreteModel(domain, partition)

        Dirichlet_tags = [5, 6, 7, 8]
        op = poisson_solver(model, f0, g0, h0, Dirichlet_tags;
                            pdegree=pdegree,
                            qdegree=qdegree)

        
        ls = LUSolver()
        solver = LinearFESolver(ls)

        wh = solve(solver, op)       
        uh, ph = wh

        Ω = get_triangulation(uh)
        dΩ = Measure(Ω, 2*pdegree+1)

        eu = u0 - uh
        error_u = sqrt(sum( ∫( eu*eu + ∇(eu)⋅∇(eu) )*dΩ ))
        append!(errors_u, error_u)

        Γ = get_triangulation(ph)
        dΓ = Measure(Γ, 2*qdegree+1)
        # On surface we should be approximating ∇u⋅ν
        ν = get_normal_vector(Γ)

        error_p = sqrt(sum(∫((ph*ν)⋅(ph*ν))*dΓ))
        append!(errors_p, error_p)

        append!(hs, 1/n)
        append!(ndofsV, length(get_free_dof_ids(uh.fe_space)))
        append!(ndofsQ, length(get_free_dof_ids(ph.fe_space)))

        !isempty(whs) && pop!(whs)
        push!(whs, wh)
    end
    wh, = whs
    uh, ph = wh

    u0h = interpolate_everywhere(u0, uh.fe_space)
    writevtk(get_triangulation(uh), "poisson_Usol", order=1, cellfields=["uh" => uh, "u" => u0h])
    writevtk(get_triangulation(ph), "poisson_Psol", order=1, cellfields=["ph" => ph])

    rates_u = log.(errors_u[2:end]./errors_u[1:end-1])./log.(hs[2:end]./hs[1:end-1])
    rates_u = [NaN; rates_u]

    rates_p = log.(errors_p[2:end]./errors_p[1:end-1])./log.(hs[2:end]./hs[1:end-1])
    rates_p = [NaN; rates_p]
    
    table = hcat(hs, ndofsV, errors_u, rates_u, ndofsQ, errors_p, rates_p)
    for row in eachrow(table)
        @printf "h = %.2E | dim(V) = %d |u-uh|_1 = %.2E rate = %.2f | dim(Q) = %d |p-ph|_0 = %.2E rate = %.2f\n" row...
    end
end