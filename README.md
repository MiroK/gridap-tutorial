# My Gridap.jl tutorial

My attempt to pick up [Gripap.jl](https://github.com/gridap) by implementing some of the [FEniCS_ii](https://github.com/MiroK/fenics_ii) 
demos, i.e. multiphysics problems with interface coupling.

## TODO multiphysics
- [x] Poisson and Poisson-Babuska
- [ ] 2d-1d problem from paper with Magne
- [ ] Darcy and Darcy-Babuska
- [ ] Stokes and Stokes-Babuska
- [ ] Marcela Szopos problem
- [ ] Stokes-Darcy free of LM
- [ ] Stokes-Darcy with LM (Layton)
- [ ] Simple EMI

## TODO infrastructure
- [ ] Getting the matrix 
- [ ] Getting preconditioner
- [ ] Setup for iterative solvers (in PETSc)
- [ ] Setup for eigenvalue computations (fractional guys)


## TODO other
- [ ] Equivalent of `tabulate_dof_coordinates`
- [ ] (Maybe) avoid interpolation when making distance functions