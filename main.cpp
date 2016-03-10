// Copyright (C) 2015 Stefano Zampini and Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2015-08-24
// Last changed: 2015-08-24
//
// This demo solves the div-div model problem
// -\nabla (\div u) + u = f
//

#include <dolfin.h>
#include "DivDiv.h"
#include <chrono>

using namespace dolfin;

int main(int argc, char* argv[])
{

  if (argc != 5)
	{
		printf("Error: Wrong number of inputs \n");
		return 0;
	}

  // BDDC solver
  class BDDCSolver
  {
  public:

    BDDCSolver(const FunctionSpace& V, const PETScMatrix& A) : _A(A)
    {
      // Test if we are working with lowest-order RT elements
      is_RT0 = !strcmp("FiniteElement('Raviart-Thomas', Domain(Cell('tetrahedron', 3)), 1, None)",
                       V.element()->signature().c_str());

      // In the case of RT0, a primal space for the BDDC is known from
      // the literature (D.-S. Oh, O. B. Widlund, and C. R. Dohrmann,
      // 2014)
      if (is_RT0)
      {
        // Create PETSc vector to hold quadrature weights representing
        // normal fluxes on subdomain faces (+1 and -1 entries
        // depending on face orientations)
        Vec quad_vec;
        MatCreateVecs(A.mat(), &quad_vec, NULL);

        // Get local to global map used to insert values in a local
        // subdomain-wise ordering of dofs
        ISLocalToGlobalMapping rmap;
        MatGetLocalToGlobalMapping(A.mat(), &rmap, NULL);
        VecSetLocalToGlobalMapping(quad_vec, rmap);
        VecSet(quad_vec, 0.0);

        // Vectors to stash the quadrature weights before calling
        // VecSetValuesLocal
        std::vector<PetscScalar> vals;
        std::vector<PetscInt> idxs;

        // Get mesh
        auto mesh = V.mesh();

        // Topological dimension
        const std::size_t tdim = mesh->topology().dim();

        // Since we are going to iterate on mesh entities, we need a
        // map from facets to the correspoing dof (in local ordering)
        std::vector<dolfin::la_index> facets_to_ldofs
          = V.dofmap()->dofs(*mesh, tdim - 1);

        // Now loop on shared facets and compute the quadrature
        // weights
        const std::unordered_map<unsigned int,
                           std::vector<std::pair<unsigned int, unsigned int>>>
          shared_facets
          = DistributedMeshTools::compute_shared_entities(*mesh, tdim - 1);
        for (auto shared_facet = shared_facets.begin();
             shared_facet != shared_facets.end(); ++shared_facet)
        {
          PetscScalar val;

          // Get facet entity
          const Facet facet(*mesh, shared_facet->first);

          // Get cell to which the current shared facet belongs
          const Cell cell(*mesh, facet.entities(tdim)[0]);

          // Get local facet index in the cell
          int lind = 0;
          for (FacetIterator facet_cell(cell); !facet_cell.end(); ++facet_cell)
          {
            if (facet_cell->index() == facet.index())
              break;
            lind++;
          }

          // Infer the orientation
          bool n_is_outward;
          if (tetrahedron_facet_orientations[lind] == 1)
            n_is_outward = true;
          else
            n_is_outward = false;

          if (!cell.orientation())
            n_is_outward = !n_is_outward;

          // Assign a unique orientation (outward from lower rank to
          // higher rank)
          const std::vector<std::pair<unsigned int, unsigned int>>&
            facet_sharing_map = shared_facet->second;
          if (PetscGlobalRank < facet_sharing_map[0].first)
          {
            if (n_is_outward)
              val = 1.0;
            else
              val = -1.0;
          }
          else
          {
            if (n_is_outward)
              val = -1.0;
            else
              val = 1.0;
          }

          // Stash the values and the corresponding indices (in dofs
          // local ordering)
          idxs.push_back(facets_to_ldofs[shared_facet->first]);
          vals.push_back(val);
        }

        // Set values in PETSc vector and assemble
        VecSetValuesLocal(quad_vec, idxs.size(), idxs.data(), vals.data(),
                          INSERT_VALUES);
        VecAssemblyBegin(quad_vec);
        VecAssemblyEnd(quad_vec);

        // Create NullSpace object for later use
        MatNullSpaceCreate(mesh->mpi_comm(), PETSC_FALSE, 1, &quad_vec,
                           &near_null_space);
        VecDestroy(&quad_vec);
      }
    }

    ~BDDCSolver()
    {
      // Destroy PETSc objects
      if (is_RT0)
        MatNullSpaceDestroy(&near_null_space);
    }

    void solve(const PETScVector& b, PETScVector& x)
    {
      // Krylov solver
      KSP ksp;
      KSPCreate(_A.mpi_comm(), &ksp);
      KSPSetType(ksp, KSPCG);
      KSPSetTolerances(ksp, 1.0e-8, 1.0e-12, 1.0e10, 1000);

      // Set operators in Krylov solver
      KSPSetOperators(ksp, _A.mat(), _A.mat());

      // Set preconditioner type to BDDC
      PC pc;
      KSPGetPC(ksp, &pc);
      PCSetType(pc, PCBDDC);

      // Customize BDDC solver
      // Since we are dealing with irregular mesh partitioning, it is
      // better to use deluxe scaling (set by options)
      PETScOptions::set("pc_bddc_use_deluxe_scaling", "true");
//      PETScOptions::set("pc_bddc_levels",4);

      PETScOptions::set("pc_bddc_coarse_redistribute",8);

      // Since RT or BDM elements don't have any degree of freedom
      // associated with the vertices of the elements, we have to
      // inform BDDC about not using the connectivity graph of local
      // degrees of freedom when analyzing the interface
      PETScOptions::set("pc_bddc_use_local_mat_graph", "false");

      // Next we customize the BDDC primal space. In the case of
      // constant coefficients with lowest Raviart-Thomas vector
      // fields, we just need to impose continuity of the normal
      // fluxes at the interface between subdomains and this could be
      // done by attaching a MatNullSpace object to the
      // preconditioning matrix in PETSc via MatSetNearNullSpace
      if (is_RT0)
        MatSetNearNullSpace(_A.mat(), near_null_space);
      else
      {
        // For BDM (any order) or RT (order > 1) the choice of primal
        // space is not known In this case we can use the adaptive
        // version of BDDC (works with symmetric positive definite
        // problems only)
//        PETScOptions::set("pc_bddc_adaptive_threshold", "10");
//        PETScOptions::set("pc_bddc_adaptive_nmin", "1");
      }

      // Setup Krylov solver
      PETScOptions::set("ksp_monitor_true_residual");
      PETScOptions::set("ksp_norm_type","natural");
//      PETScOptions::set("ksp_view");
      KSPSetFromOptions(ksp);
      KSPSetUp(ksp);
      PETScOptions::set("pc_factor_mat_solver_package","mumps");

      // Solve
      KSPSolve(ksp, b.vec(), x.vec());
      KSPDestroy(&ksp);

      // Update ghosts
      x.apply("insert");
    }

  private:

    const PETScMatrix& _A;
    MatNullSpace near_null_space;
    bool is_RT0;
  };

  // Homogenous forcing term
  class Source : public Expression
  {
  public:

    Source() : Expression(3) {}

    void eval(Array<double>& values, const Array<double>& x) const
    {
      values[0] = 0.0;
      values[1] = 0.0;
      values[2] = 1.0;
    }
  };

  // Zero Dirichlet BC
  class Zero : public Expression
  {
  public:

    Zero() : Expression(3) {}

    void eval(Array<double>& values, const Array<double>& x) const
    {
      values[0] = 0.0;
      values[1] = 0.0;
      values[2] = 0.0;
    }
  };

  // Everywhere on external surface
  class DirichletBoundary: public SubDomain
  {
    bool inside(const Array<double>& x, bool on_boundary) const
    { return on_boundary; }
  };

  // Load sphere mesh
  //auto mesh = std::make_shared<Mesh>("../sphere.xml.gz");
  auto mesh = std::make_shared<Mesh>(UnitCubeMesh(atoi(argv[1]),atoi(argv[2]),atoi(argv[3])));

  // Define functions
//  auto f = std::make_shared<Source>();
  auto zero = std::make_shared<Constant>(0);
  auto f = std::make_shared<Constant>(1);
  auto k = std::make_shared<Constant>(atoi(argv[4]));
  auto g = std::make_shared<Constant>(0);

  // Define function space and boundary condition
  auto V = std::make_shared<DivDiv::FunctionSpace>(mesh);
  auto boundary = std::shared_ptr<DirichletBoundary>(new DirichletBoundary);
  auto bc = std::make_shared<DirichletBC>(V, zero, boundary);

  // Define variational problem
  DivDiv::BilinearForm a(V, V);
  DivDiv::LinearForm L(V);
  L.f = f;
  a.k = k;
  L.g = g;

  // Compute solution
  Function T(V);
  bool test_BDDC = true;
  if (MPI::size(mesh->mpi_comm()) == 1)
  {
    std::cout << "Using default linear solver: BDDC is not intended to be run sequentially"
              << std::endl;
    test_BDDC = false;
  }

  if (!test_BDDC)
  {
    // Solve problem using default solver
    solve(a == L, T, *bc);
  }
  else
  {
    PETScMatrix A;
    PETScVector b;

    // Assemble into PETSc "is" matrix type
    PETScOptions::set("mat_type", "is");
    assemble_system(A, b, a, L, {bc});

    // Create solver and solve
    PETScVector& x = as_type<PETScVector>(*T.vector());
    BDDCSolver bddc(*V, A);
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    bddc.solve(b, x);
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();

    printf("%f \n", std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()/1000000.0);
  }

  // Dump solution
//  File file("output.pvd");
//  file << T;

  // Plot solution
  //plot(T);
  //interactive();

  return 0;
}
