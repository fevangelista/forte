/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2025 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#pragma once

#include <fstream>
#include <iomanip>

#include "sci/sci.h"
#include "sparse_ci/sparse_ci_solver.h"
#include "helpers/timer.h"

using d1 = std::vector<double>;
using d2 = std::vector<d1>;

namespace forte {

class Reference;

enum class AverageFunction { MaxF, AvgF };

/**
 * @brief The AdaptiveCI class
 * This class implements an adaptive CI algorithm
 */
class AdaptiveCI : public SelectedCIMethod {
  public:
    // ==> Class Constructor and Destructor <==

    /**
     * Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info A pointer to the MOSpaceInfo object
     */
    AdaptiveCI(StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
               std::shared_ptr<ForteOptions> options, std::shared_ptr<MOSpaceInfo> mo_space_info,
               std::shared_ptr<ActiveSpaceIntegrals> as_ints);

    // ==> Class Interface <==

    /// Set options from an option object
    /// @param options the options passed in
    void set_options(std::shared_ptr<ForteOptions>) override {}

    // Interfaces of SCI algorithm
    /// Print the banner and starting information.
    void print_info() override;
    /// Pre-iter preparation, usually includes preparing an initial reference
    void pre_iter_preparation() override;
    /// Step 1. Diagonalize the Hamiltonian in the P space
    void diagonalize_P_space() override;
    /// Step 2. Find determinants in the Q space
    void find_q_space() override;
    /// Step 3. Diagonalize the Hamiltonian in the P + Q space
    void diagonalize_PQ_space() override;
    /// Step 4. Check convergence
    bool check_convergence() override;
    /// Step 5. Prune the P + Q space to get an updated P space
    void prune_PQ_to_P() override;
    /// Post-iter process
    void post_iter_process() override;
    /// Full PT2 correction
    void full_mrpt2();

    // Temporarily added interface to ExcitedStateSolver
    /// Set the class variable
    void set_method_variables(
        std::string ex_alg, size_t nroot_method, size_t root,
        const std::vector<std::vector<std::pair<Determinant, double>>>& old_roots) override;
    /// Getters
    DeterminantHashVec get_PQ_space() override;
    std::shared_ptr<psi::Matrix> get_PQ_evecs() override;
    std::shared_ptr<psi::Vector> get_PQ_evals() override;
    std::vector<double> get_PQ_spin2() override;
    size_t get_ref_root() override;
    std::vector<double> get_multistate_pt2_energy_correction() override;

    /// Set the printing level
    void set_quiet(bool quiet) { quiet_mode_ = quiet; }

    /// Compute the ACI-NOs
    void print_nos();

    void semi_canonicalize();
    void set_fci_ints(std::shared_ptr<ActiveSpaceIntegrals> fci_ints);

    void upcast_reference(DeterminantHashVec& ref);

    // Update sigma
    void update_sigma();

  private:
    // Temporarily added
    std::shared_ptr<psi::Matrix> P_evecs_;
    std::shared_ptr<psi::Vector> P_evals_;
    DeterminantHashVec P_space_;
    DeterminantHashVec P_ref_;
    std::vector<double> P_ref_evecs_;
    std::vector<double> P_energies_;
    std::vector<std::vector<double>> energy_history_;
    int num_ref_roots_;
    bool follow_;
    local_timer cycle_time_;

    // Temporarily added interface to ExcitedStateSolver
    std::shared_ptr<psi::Matrix> PQ_evecs_;
    std::shared_ptr<psi::Vector> PQ_evals_;
    DeterminantHashVec PQ_space_;
    std::vector<double> PQ_spin2_;

    // ==> Class data <==

    /// The reference determinant
    std::vector<Determinant> initial_reference_;
    /// The PT2 energy correction
    std::vector<double> multistate_pt2_energy_correction_;
    bool set_ints_ = false;

    // ==> ACI Options <==
    /// The threshold applied to the primary space
    double sigma_;
    /// The threshold applied to the secondary space
    double gamma_;
    /// The prescreening threshold
    double screen_thresh_;
    /// Use threshold from perturbation theory?
    bool perturb_select_;

    /// Add missing degenerate determinants excluded from the aimed selection?
    bool add_aimed_degenerate_;

    /// The function used in the averaged multiroot algorithm to find the q-space
    AverageFunction average_function_;
    /// the q reference
    std::string q_reference_;
    /// Algorithm for computing excited states
    std::string ex_alg_;
    /// The reference root
    int ref_root_;
    /// The reference root
    int root_;
    /// Enable aimed selection
    bool aimed_selection_;
    /// If true select by energy, if false use first-order coefficient
    bool energy_selection_;
    /// Number of roots to calculate for final excited state
    int post_root_;
    /// Rediagonalize H?
    bool post_diagonalize_;
    /// Print warning?
    bool print_warning_;
    /// Spin tolerance
    double spin_tol_;
    /// Compute 1-RDM?
    bool compute_rdms_;
    /// Print a determinant analysis?
    bool det_hist_;
    /// Save dets to file?
    bool det_save_;
    /// The number of states to average
    int naverage_;
    /// An offset for averaging states
    int average_offset_;

    /// Control streamlining
    bool streamline_qspace_;
    /// The CI coeffiecients
    std::shared_ptr<psi::Matrix> evecs_;

    bool build_lists_;

    /// A map of determinants in the P space
    std::unordered_map<Determinant, int, Determinant::Hash> P_space_map_;
    /// A History of Determinants
    std::unordered_map<Determinant, std::vector<std::pair<size_t, std::string>>, Determinant::Hash>
        det_history_;
    /// Stream for printing determinant coefficients
    std::ofstream det_list_;
    /// Roots to project out
    std::vector<std::vector<std::pair<size_t, double>>> bad_roots_;
    /// Storage of past roots
    std::vector<std::vector<std::pair<Determinant, double>>> old_roots_;

    /// Form initial guess space with correct spin? ****OBSOLETE?*****
    bool do_guess_;
    /// Spin-symmetrized evecs
    std::shared_ptr<psi::Matrix> PQ_spin_evecs_;
    /// The unselected part of the SD space
    det_hash<double> external_wfn_;
    /// Do approximate RDM?
    bool approx_rdm_ = false;

    bool print_weights_;

    /// The alpha MO always unoccupied
    size_t hole_;

    /// Whether iteration is within gas space
    bool gas_iteration_;

    /// Whether an occupation analysis is ran after
    bool occ_analysis_;

    /// Timing variables
    double build_H_;
    double diag_H_;
    double build_space_;
    double screen_space_;
    double spin_trans_;

    // ==> Class functions <==

    /// All that happens before we compute the energy
    void startup();

    /// Generate set of state-averaged q-criteria and determinants
    double average_q_values(const std::vector<double>& E2);

    /// Get criteria for a specific root
    double root_select(int nroot, std::vector<double>& C1, std::vector<double>& E2);

    /// Basic determinant generator (threaded, no batching, all determinants stored)
    void get_excited_determinants_avg(int nroot, std::shared_ptr<psi::Matrix> evecs,
                                      std::shared_ptr<psi::Vector> evals,
                                      DeterminantHashVec& P_space,
                                      std::vector<std::pair<double, Determinant>>& F_space);

    /// Get excited determinants with a specified hole
    //  void get_excited_determinants_restrict(int nroot, std::shared_ptr<psi::Matrix> evecs,
    //  std::shared_ptr<psi::Vector> evals,  DeterminantHashVec& P_space,
    //                                         std::vector<std::pair<double, Determinant>>&
    //                                         F_space);
    /// Get excited determinants with a specified hole
    void get_excited_determinants_core(std::shared_ptr<psi::Matrix> evecs,
                                       std::shared_ptr<psi::Vector> evals,
                                       DeterminantHashVec& P_space,
                                       std::vector<std::pair<double, Determinant>>& F_space);

    // Optimized for a single root
    void get_excited_determinants_sr(std::shared_ptr<psi::Matrix> evecs,
                                     std::shared_ptr<psi::Vector> evals,
                                     DeterminantHashVec& P_space,
                                     std::vector<std::pair<double, Determinant>>& F_space);

    // (DEFAULT in batching) Optimized batching algorithm, prescreens the batches to significantly
    // reduce storage, based on hashes
    double get_excited_determinants_batch(std::shared_ptr<psi::Matrix> evecs,
                                          std::shared_ptr<psi::Vector> evals,
                                          DeterminantHashVec& P_space,
                                          std::vector<std::pair<double, Determinant>>& F_space);

    // Gets excited determinants using sorting of vectors
    double get_excited_determinants_batch_vecsort(
        std::shared_ptr<psi::Matrix> evecs, std::shared_ptr<psi::Vector> evals,
        DeterminantHashVec& P_space, std::vector<std::pair<double, Determinant>>& F_space);

    // Optimized for a single root, in GAS
    void get_gas_excited_determinants_sr(std::shared_ptr<psi::Matrix> evecs,
                                         std::shared_ptr<psi::Vector> evals,
                                         DeterminantHashVec& P_space,
                                         std::vector<std::pair<double, Determinant>>& F_space);

    /// Basic determinant generator (threaded, no batching, all determinants stored), in GAS
    void get_gas_excited_determinants_avg(int nroot, std::shared_ptr<psi::Matrix> evecs,
                                          std::shared_ptr<psi::Vector> evals,
                                          DeterminantHashVec& P_space,
                                          std::vector<std::pair<double, Determinant>>& F_space);

    void get_gas_excited_determinants_core(std::shared_ptr<psi::Matrix> evecs,
                                           std::shared_ptr<psi::Vector> evals,
                                           DeterminantHashVec& P_space,
                                           std::vector<std::pair<double, Determinant>>& F_space);

    /// (DEFAULT)  Builds excited determinants for a bin, uses all threads, hash-based
    det_hash<double> get_bin_F_space(int bin, int nbin, double E0,
                                     std::shared_ptr<psi::Matrix> evecs,
                                     DeterminantHashVec& P_space);

    /// Builds core excited determinants for a bin, uses all threads, hash-based
    det_hash<double> get_bin_F_space_core(int bin, int nbin, double E0,
                                          std::shared_ptr<psi::Matrix> evecs,
                                          DeterminantHashVec& P_space);
    /// Builds excited determinants in batch using sorting of vectors
    std::pair<std::vector<std::vector<std::pair<Determinant, double>>>, std::vector<size_t>>
    get_bin_F_space_vecsort(int bin, int nbin, std::shared_ptr<psi::Matrix> evecs,
                            DeterminantHashVec& P_space);

    /// Prune the space of determinants
    void prune_q_space(DeterminantHashVec& PQ_space, DeterminantHashVec& P_space,
                       std::shared_ptr<psi::Matrix> evecs);

    /// Check if the procedure has converged
    bool check_convergence(std::vector<std::vector<double>>& energy_history,
                           std::shared_ptr<psi::Vector> new_energies);

    /// Check if the procedure is stuck
    bool check_stuck(const std::vector<std::vector<double>>& energy_history,
                     std::shared_ptr<psi::Vector> evals);

    /// Compute overlap for root following
    int root_follow(DeterminantHashVec& P_ref, std::vector<double>& P_ref_evecs,
                    DeterminantHashVec& P_space, std::shared_ptr<psi::Matrix> P_evecs,
                    int num_ref_roots);

    /// Add roots to be projected out in DL
    void add_bad_roots(DeterminantHashVec& dets);

    /// Set PT2 energy correction to zero;
    void zero_multistate_pt2_energy_correction();

    /// Print GAS information
    void print_gas_wfn(DeterminantHashVec& space, std::shared_ptr<psi::Matrix> evecs);

    /// Print occ number
    void print_occ_number(DeterminantHashVec& space, std::shared_ptr<psi::Matrix> evecs);

    /// number of GAS
    size_t gas_num_;

    /// Allowed single excitation from one GAS to another
    std::pair<std::map<std::vector<int>, std::vector<std::pair<size_t, size_t>>>,
              std::map<std::vector<int>, std::vector<std::pair<size_t, size_t>>>>
        gas_single_criterion_;

    /// Allowed double excitation from two GAS to another two GAS
    std::tuple<std::map<std::vector<int>, std::vector<std::tuple<size_t, size_t, size_t, size_t>>>,
               std::map<std::vector<int>, std::vector<std::tuple<size_t, size_t, size_t, size_t>>>,
               std::map<std::vector<int>, std::vector<std::tuple<size_t, size_t, size_t, size_t>>>>
        gas_double_criterion_;

    /// Electron configurations
    std::vector<std::vector<int>> gas_electrons_;

    /// Relative mo in the entire active space for each GAS;
    std::vector<std::vector<size_t>> relative_gas_mo_;
};

} // namespace forte
