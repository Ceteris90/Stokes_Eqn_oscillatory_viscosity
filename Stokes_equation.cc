/*Including the header files*/
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/generic_linear_algebra.h>
/* #define FORCE_USE_OF_TRILINOS */
namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

//STL
#include <cmath>
#include <fstream>
#include <iostream>
namespace Step55
{
  using namespace dealii;
  namespace LinearSolvers
  {
    template <class Matrix, class Preconditioner>
    class InverseMatrix : public Subscriptor
    {
    public:
      InverseMatrix(const Matrix &m, const Preconditioner &preconditioner);
      template <typename VectorType>
      void vmult(VectorType &dst, const VectorType &src) const;
    private:
      const SmartPointer<const Matrix> matrix;
      const Preconditioner &           preconditioner;
    };
    
    template <class Matrix, class Preconditioner>
    InverseMatrix<Matrix, Preconditioner>::InverseMatrix(
      const Matrix &        m,
      const Preconditioner &preconditioner)
      : matrix(&m)
      , preconditioner(preconditioner)
    {}
    
    template <class Matrix, class Preconditioner>
    template <typename VectorType>
    void
    InverseMatrix<Matrix, Preconditioner>::vmult(VectorType &      dst,
                                                 const VectorType &src) const
    {
      SolverControl solver_control(src.size(), 1e-8 * src.l2_norm());
      SolverCG<LA::MPI::Vector> cg(solver_control);
      dst = 0;
      try
        {
          cg.solve(*matrix, dst, src, preconditioner);
        }
      catch (std::exception &e)
        {
          Assert(false, ExcMessage(e.what()));
        }
    }
    
    template <class PreconditionerA, class PreconditionerS>
    class BlockDiagonalPreconditioner : public Subscriptor
    {
    public:
      BlockDiagonalPreconditioner(
      const PreconditionerA &preconditioner_A,
                                  const PreconditionerS &preconditioner_S);
      void vmult(LA::MPI::BlockVector &      dst,
                 const LA::MPI::BlockVector &src) const;
    private:
      const PreconditionerA &preconditioner_A;
      const PreconditionerS &preconditioner_S;
    };
    
    template <class PreconditionerA, class PreconditionerS>
    BlockDiagonalPreconditioner<PreconditionerA, PreconditionerS>::
      BlockDiagonalPreconditioner(const PreconditionerA &preconditioner_A,
                                  const PreconditionerS &preconditioner_S)
      : preconditioner_A(preconditioner_A)
      , preconditioner_S(preconditioner_S)
    {}
    
    template <class PreconditionerA, class PreconditionerS>
    void BlockDiagonalPreconditioner<PreconditionerA, PreconditionerS>::vmult(
      LA::MPI::BlockVector &      dst,
      const LA::MPI::BlockVector &src) const
    {
      preconditioner_A.vmult(dst.block(0), src.block(0));
      preconditioner_S.vmult(dst.block(1), src.block(1));
    }
  } // namespace LinearSolvers
  
  
  template <int dim>
  class StokesBoundaryValues : public Function<dim>
  {
  public:
    StokesBoundaryValues()
      : Function<dim>(dim + 1)
    {}
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &  value) const override;
    
  };
  
  template <int dim>
  double StokesBoundaryValues<dim>::value(const Point<dim> & /*p*/,
                                          const unsigned int component) const
  {
      
    Assert(component < this->n_components,
           ExcIndexRange(component, 0, this->n_components));
    return 0;

  }
  
  template <int dim>
  void StokesBoundaryValues<dim>::vector_value(const Point<dim> &p,
                                               Vector<double> &  values) const
  {
    for (unsigned int c = 0; c < this->n_components; ++c)
      values(c) = StokesBoundaryValues<dim>::value(p, c);
  }

  
  
  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide()
      : Function<dim>(dim + 1)
    {}
    virtual void vector_value(const Point<dim> & p,
                              Vector<double> &  value) const override;
  };
  
  template <int dim>
  void RightHandSide<dim>::vector_value(const Point<dim> & p,
                                        Vector<double> &  values) const
  {
   //Case 1: First Test Problem Kovaszmay Flow//
  // ///////////////////////////////////////////


    const double R_x = p[0];
    const double R_y = p[1];
    const double pi  = numbers::PI;
    const double pi2 = pi * pi;
    values[0] =
      -1.0L / 2.0L * (-2 * sqrt(25.0 + 4 * pi2) + 10.0) *
        exp(R_x * (-2 * sqrt(25.0 + 4 * pi2) + 10.0)) -
      0.4 * pi2 * exp(R_x * (-sqrt(25.0 + 4 * pi2) + 5.0)) * cos(2 * R_y * pi) +
      0.1 * pow(-sqrt(25.0 + 4 * pi2) + 5.0, 2) *
        exp(R_x * (-sqrt(25.0 + 4 * pi2) + 5.0)) * cos(2 * R_y * pi);
    values[1] = 0.2 * pi * (-sqrt(25.0 + 4 * pi2) + 5.0) *
                  exp(R_x * (-sqrt(25.0 + 4 * pi2) + 5.0)) * sin(2 * R_y * pi) -
                0.05 * pow(-sqrt(25.0 + 4 * pi2) + 5.0, 3) *
                  exp(R_x * (-sqrt(25.0 + 4 * pi2) + 5.0)) * sin(2 * R_y * pi) /
                  pi;
    values[2] = 0;
    
	  
   //Case 2:  Second Test Problem Wang Flow //
    /////////////////////////////////////

    
    const double pi  = numbers::PI;

    values[0] = 1-cos(10*p[0])*exp(-10*p[1]);
    values[1] = (10/pi)*sin(10*p[0])*exp(-10*p[1]);
    values[2] = 0;
  }
  
  template <int dim>
  class Viscosity: public Function<dim>
  {
   public:
    
     virtual double value(const Point<dim> &p,const unsigned int component=0) const override;
  };
  
  template <int dim>
  double Viscosity<dim>::value(const Point<dim> &p,const unsigned int component) const
  {

	  const double pi=numbers::PI;
    constexpr double v_min = 0.01;
    constexpr double beta = 0;
  	const int k = 50;
  	return v_min*( 1+ beta*sin(2*k*pi*p[0])*cos(2*k*pi*p[1]));
	
  }


  template <int dim>
  class StokesProblem
  {
  public:
    StokesProblem(unsigned int velocity_degree);
    void run();
  private:
    void make_grid();
    void setup_system();
    void assemble_system();
    void solve();
    void refine_grid();
    void output_results(const unsigned int cycle) const;
    
    unsigned int velocity_degree;
    double       viscosity;
    
    MPI_Comm     mpi_communicator;
    
    FESystem<dim>                             fe;
    parallel::distributed::Triangulation<dim> triangulation;
    DoFHandler<dim>                           dof_handler;
    
    std::vector<IndexSet> owned_partitioning;
    std::vector<IndexSet> relevant_partitioning;
    
    AffineConstraints<double> constraints;
    
    LA::MPI::BlockSparseMatrix system_matrix;
    LA::MPI::BlockSparseMatrix preconditioner_matrix;
    LA::MPI::BlockVector       locally_relevant_solution;
    LA::MPI::BlockVector       system_rhs;
    
    ConditionalOStream pcout;
    
    TimerOutput        computing_timer;
    
  };
  
  template <int dim>
  StokesProblem<dim>::StokesProblem(unsigned int velocity_degree)
    : velocity_degree(velocity_degree)
    , mpi_communicator(MPI_COMM_WORLD)
    , fe(FE_Q<dim>(velocity_degree), dim, FE_Q<dim>(velocity_degree - 1), 1)
    , triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing(
                      Triangulation<dim>::smoothing_on_refinement |
                      Triangulation<dim>::smoothing_on_coarsening))
    , dof_handler(triangulation)
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
  {}
  
  
  template <int dim>
  void StokesProblem<dim>::make_grid()
  {
    GridGenerator::hyper_cube(triangulation, -0.5, 1.5);
    triangulation.refine_global(3);

  }
  
  
  template <int dim>
  void StokesProblem<dim>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "setup");
    
    dof_handler.distribute_dofs(fe);
    
    std::vector<unsigned int> stokes_sub_blocks(dim + 1, 0);
    stokes_sub_blocks[dim] = 1;
    DoFRenumbering::component_wise(dof_handler, stokes_sub_blocks);
    
    const std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler, stokes_sub_blocks);
      
    const unsigned int n_u = dofs_per_block[0];
    const unsigned int n_p = dofs_per_block[1];
    
    pcout << "   Number of degrees of freedom: " << dof_handler.n_dofs() << " ("
          << n_u << '+' << n_p << ')' << std::endl;
    owned_partitioning.resize(2);
    owned_partitioning[0] = dof_handler.locally_owned_dofs().get_view(0, n_u);
    owned_partitioning[1] = dof_handler.locally_owned_dofs().get_view(n_u, n_u + n_p);
      
    
    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    relevant_partitioning.resize(2);
    
    relevant_partitioning[0] = locally_relevant_dofs.get_view(0, n_u);
    relevant_partitioning[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);
    {
      constraints.reinit(locally_relevant_dofs);
      
      FEValuesExtractors::Vector velocities(0);
      
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               StokesBoundaryValues<dim>(),
                                               constraints,
                                               fe.component_mask(velocities));
     
      VectorTools::interpolate_boundary_values(
                                               dof_handler,
                                               1,
                                               StokesBoundaryValues<dim>(),
                                               constraints,
                                               fe.component_mask(velocities));                                   
                                               
                                                                
      constraints.close();
    }
    {
      TrilinosWrappers::BlockSparsityPattern bsp(owned_partitioning,
                                                 owned_partitioning,
                                                 relevant_partitioning,
                                                 mpi_communicator);
                                                 
      Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
      for (unsigned int c = 0; c < dim + 1; ++c)
        for (unsigned int d = 0; d < dim + 1; ++d)
          if (!((c == dim) && (d == dim)))
            coupling[c][d] = DoFTools::always;
          else
            coupling[c][d] = DoFTools::none;
            
      DoFTools::make_sparsity_pattern(dof_handler,
                                      coupling,
                                      bsp,
                                      constraints,
                                      false,
                                      Utilities::MPI::this_mpi_process(
                                        mpi_communicator));
      bsp.compress();
      system_matrix.reinit(bsp);
    }
    {
      TrilinosWrappers::BlockSparsityPattern preconditioner_bsp(
        owned_partitioning,
        owned_partitioning,
        relevant_partitioning,
        mpi_communicator);
        
      Table<2, DoFTools::Coupling> preconditioner_coupling(dim + 1, dim + 1);
      for (unsigned int c = 0; c < dim + 1; ++c)
        for (unsigned int d = 0; d < dim + 1; ++d)
          if ((c == dim) && (d == dim))
            preconditioner_coupling[c][d] = DoFTools::always;
          else
            preconditioner_coupling[c][d] = DoFTools::none;
            
      DoFTools::make_sparsity_pattern(dof_handler,
                                      preconditioner_coupling,
                                      preconditioner_bsp,
                                      constraints,
                                      false,
                                      Utilities::MPI::this_mpi_process(
                                        mpi_communicator));
      preconditioner_bsp.compress();
      
      preconditioner_matrix.reinit(preconditioner_bsp);
    }
    locally_relevant_solution.reinit(owned_partitioning,
                                     relevant_partitioning,
                                     mpi_communicator);
    system_rhs.reinit(owned_partitioning, mpi_communicator);
  }
  
  
  
  template <int dim>
  void StokesProblem<dim>::assemble_system()
  {
    TimerOutput::Scope t(computing_timer, "assembly");
    system_matrix         = 0;
    preconditioner_matrix = 0;
    system_rhs            = 0;
    const QGauss<dim> quadrature_formula(velocity_degree + 1);
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_matrix2(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
    
    Viscosity<dim>  viscosity;
    std::vector<double> viscosity_values(n_q_points);
    
    const RightHandSide<dim>    right_hand_side;
    std::vector<Vector<double>> rhs_values(n_q_points, Vector<double>(dim + 1));
    
    std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
    std::vector<double>         div_phi_u(dofs_per_cell);
    std::vector<double>         phi_p(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    
    const FEValuesExtractors::Vector     velocities(0);
    const FEValuesExtractors::Scalar     pressure(dim);
    
   
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_matrix  = 0;
          cell_matrix2 = 0;
          cell_rhs     = 0;
          fe_values.reinit(cell);
          
          viscosity.value_list(fe_values.get_quadrature_points(),
                                   viscosity_values);
          
          right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
                                            rhs_values);
                                            
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                  grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                  div_phi_u[k]  = fe_values[velocities].divergence(k, q);
                  phi_p[k]      = fe_values[pressure].value(k, q);
                }
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      cell_matrix(i, j) +=
                        (viscosity_values[q] *
                           scalar_product(grad_phi_u[i], grad_phi_u[j]) -
                         div_phi_u[i] * phi_p[j] - phi_p[i] * div_phi_u[j]) *
                        fe_values.JxW(q);
                      cell_matrix2(i, j) += 1/viscosity_values[q]* phi_p[i] * 
                                            phi_p[j] * fe_values.JxW(q);
                    }
                  const unsigned int component_i =
                    fe.system_to_component_index(i).first;
                  cell_rhs(i) += fe_values.shape_value(i, q) *
                                 rhs_values[q](component_i) * fe_values.JxW(q);
                }
            }
          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_matrix,
                                                 cell_rhs,
                                                 local_dof_indices,
                                                 system_matrix,
                                                 system_rhs);
          constraints.distribute_local_to_global(cell_matrix2,
                                                 local_dof_indices,
                                                 preconditioner_matrix);
        }
    system_matrix.compress(VectorOperation::add);
    preconditioner_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }
  
  
  
  template <int dim>
  void StokesProblem<dim>::solve()
  {
    TimerOutput::Scope t(computing_timer, "solve");
    LA::MPI::PreconditionAMG prec_A;
    {
      LA::MPI::PreconditionAMG::AdditionalData data;
#ifdef USE_PETSC_LA
      data.symmetric_operator = true;
#endif
      prec_A.initialize(system_matrix.block(0, 0), data);
    }
    LA::MPI::PreconditionAMG prec_S;
    {
      LA::MPI::PreconditionAMG::AdditionalData data;
#ifdef USE_PETSC_LA
      data.symmetric_operator = true;
#endif
      prec_S.initialize(preconditioner_matrix.block(1, 1), data);
    }
    using mp_inverse_t = LinearSolvers::InverseMatrix<LA::MPI::SparseMatrix,
                                                      LA::MPI::PreconditionAMG>;
    const mp_inverse_t mp_inverse(preconditioner_matrix.block(1, 1), prec_S);
    const LinearSolvers::BlockDiagonalPreconditioner<LA::MPI::PreconditionAMG,
                                                     mp_inverse_t>
    	preconditioner(prec_A, mp_inverse);
    SolverControl solver_control(system_matrix.m(),
                                 1e-10 * system_rhs.l2_norm());
    SolverMinRes<LA::MPI::BlockVector> solver(solver_control);
    LA::MPI::BlockVector distributed_solution(owned_partitioning,
                                              mpi_communicator);
    constraints.set_zero(distributed_solution);
   
    solver.solve(system_matrix, distributed_solution,system_rhs,preconditioner);
    
    pcout << "   Solved in " << solver_control.last_step() << " iterations."
          << std::endl;
          
    constraints.distribute(distributed_solution);
    locally_relevant_solution = distributed_solution;
    const double mean_pressure =
      VectorTools::compute_mean_value(dof_handler,
                                      QGauss<dim>(velocity_degree + 2),
                                      locally_relevant_solution,
                                      dim);
    distributed_solution.block(1).add(-mean_pressure);
    locally_relevant_solution.block(1) = distributed_solution.block(1);
  }
  
  
  template <int dim>
  void StokesProblem<dim>::refine_grid()
  {
    TimerOutput::Scope t(computing_timer, "refine");
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    FEValuesExtractors::Vector velocities(0);
    KellyErrorEstimator<dim>::estimate(
    dof_handler,
    QGauss<dim - 1>(fe.degree + 1),
    std::map<types::boundary_id, const Function<dim> *>(),
    locally_relevant_solution,
    estimated_error_per_cell,
    fe.component_mask(velocities));
    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
    triangulation, estimated_error_per_cell, 0.3, 0.0);
    triangulation.execute_coarsening_and_refinement(); 
   }
   
   
  template <int dim>
  void StokesProblem<dim>::output_results(const unsigned int cycle) const
  {
    
    std::vector<std::string> solution_names(dim, "velocity");
    
    solution_names.emplace_back("pressure");
    
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
        
    data_component_interpretation.push_back(
    DataComponentInterpretation::component_is_scalar);
      {
  	 GridOut               grid_out;
  	 
         std::ofstream         output("grid_" + std::to_string(cycle) + ".vtu");
         
         GridOutFlags::Gnuplot gnuplot_flags(false, 5);
         
    	 grid_out.set_flags(gnuplot_flags);
    	 
         MappingQGeneric<dim> mapping(3);
         
         grid_out.write_gnuplot(triangulation, output, &mapping);
      }
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
   
    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.build_patches();
    data_out.write_vtu_with_pvtu_record(
      "./", "solution", cycle, mpi_communicator, 2);
  }
  
  
  template <int dim>
  void StokesProblem<dim>::run()
  {
#ifdef USE_PETSC_LA
    pcout << "Running using PETSc." << std::endl;
#else
    pcout << "Running using Trilinos." << std::endl;
#endif
    const unsigned int n_cycles = 5;
    for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
      {
        pcout << "Cycle " << cycle << ':' << std::endl;
        if (cycle == 0)
          make_grid();
        else
          refine_grid();
        setup_system();
        assemble_system();
        solve();
        if (Utilities::MPI::n_mpi_processes(mpi_communicator) <= 32)
          {
            TimerOutput::Scope t(computing_timer, "output");
            output_results(cycle);
          }
        computing_timer.print_summary();
        computing_timer.reset();
        pcout << std::endl;
      }
  }
} // namespace Step55


int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace Step55;
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      StokesProblem<2> problem(2);
      problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}
