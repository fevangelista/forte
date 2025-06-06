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

#include <map>
#include <vector>
#include <string>

#include <ambit/tensor.h>
#include "psi4/libmints/matrix.h"

namespace ambit {
class BlockedTensor;
}

#include "base_classes/state_info.h"
#include "base_classes/rdms.h"
#include "helpers/printing.h"

#include "sparse_ci/determinant_hashvector.h"

namespace forte {

class ActiveSpaceMethod;
class ActiveSpaceIntegrals;
class ForteIntegrals;
class ForteOptions;
class MOSpaceInfo;
class SCFInfo;
class ActiveMultipoleIntegrals;

/**
 * @class ActiveSpaceSolver
 *
 * @brief General class for a multi-state active space solver
 *
 * This class can run state-specific, multi-state, and state-averaged computations
 * on small subset of the full orbital space (<30-40 orbitals).
 */
class ActiveSpaceSolver {
  public:
    // ==> Class Constructor and Destructor <==
    /**
     * @brief ActiveSpaceMethod Constructor for a multi-state computation
     * @param solver_type A string that labels the solver requested (e.g. "FCI", "ACI", ...)
     * @param nroots_map A map of electronic states to the number of roots computed {state_1 : n_1,
     * state_2 : n_2, ...} where state_i specifies the symmetry of a state and n_i is the number of
     * levels computed.
     * @param state information about the electronic state
     * @param mo_space_info a MOSpaceInfo object
     * @param as_ints integrals for active space
     */
    ActiveSpaceSolver(const std::string& solver_type,
                      const std::map<StateInfo, size_t>& state_nroots_map,
                      std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<MOSpaceInfo> mo_space_info,
                      std::shared_ptr<ForteOptions> options,
                      std::shared_ptr<ActiveSpaceIntegrals> as_ints);

    // ==> Class Interface <==

    /// Set the print level
    const std::string& solver_type() const;

    /// @param level the print level
    void set_print(PrintLevel level);

    /// Compute the energy and return it // TODO: document (Francesco)
    const std::map<StateInfo, std::vector<double>>& compute_energy();

    /// Compute permanent dipole and quadrupole moments
    void compute_multipole_moment(std::shared_ptr<ActiveMultipoleIntegrals> ampints, int level = 1);

    /// Compute the oscillator strengths assuming same orbitals
    void compute_fosc_same_orbs(std::shared_ptr<ActiveMultipoleIntegrals> ampints);

    /// Compute the contracted CI energy
    const std::map<StateInfo, std::vector<double>>&
    compute_contracted_energy(std::shared_ptr<forte::ActiveSpaceIntegrals> as_ints,
                              int max_rdm_level);

    /// Compute RDMs of all states in the given map
    /// First entry of the pair corresponds to bra and the second is the ket.
    std::vector<std::shared_ptr<RDMs>> rdms(
        std::map<std::pair<StateInfo, StateInfo>, std::vector<std::pair<size_t, size_t>>>& elements,
        int max_rdm_level, RDMsType rdm_type);

    /// Compute a generalized RDM for a given state
    /// This will compute the quantity
    ///    R^{p1 p2 ..}_{q1 q2 ..} = X_I <Phi_I| a^+_p1 a^+_p2 .. a_q2 a_q1 |Phi_J> C_J (c_right =
    ///    true)
    /// or
    ///    R^{p1 p2 ..}_{q1 q2 ..} = C_I <Phi_I| a^+_p1 a^+_p2 .. a_q2 a_q1 |Phi_J> X_J (c_right
    ///    = false)
    void generalized_rdms(const StateInfo& state, size_t root, const std::vector<double>& X,
                          ambit::BlockedTensor& result, bool c_right, int rdm_level,
                          std::vector<std::string> spin = {});

    /// Add k-body contributions to the sigma vector
    ///    σ_I += h_{p1,p2,...}^{q1,q2,...} <Phi_I| a^+_p1 a^+_p2 .. a_q2 a_q1 |Phi_J> C_J
    /// @param state: StateInfo (symmetry, multiplicity, etc.)
    /// @param root: the root number of the state
    /// @param h: the antisymmetrized k-body integrals
    /// @param block_label_to_factor: map from the block labels of integrals to its factors
    /// @param sigma: the sigma vector to be added
    void add_sigma_kbody(const StateInfo& state, size_t root, ambit::BlockedTensor& h,
                         const std::map<std::string, double>& block_label_to_factor,
                         std::vector<double>& sigma);

    /// Compute generalized sigma vector
    ///     σ_I = <Phi_I| H |Phi_J> X_J where H is the active space Hamiltonian (fci_ints)
    /// @param state: StateInfo (symmetry, multiplicity, etc.)
    /// @param x: the X vector to be contracted with H_IJ
    /// @param sigma: the sigma vector (will be zeroed first)
    void generalized_sigma(const StateInfo& state, std::shared_ptr<psi::Vector> x,
                           std::shared_ptr<psi::Vector> sigma);

    /// Compute the state-averaged reference
    std::shared_ptr<RDMs>
    compute_average_rdms(const std::map<StateInfo, std::vector<double>>& state_weights_map,
                         int max_rdm_level, RDMsType rdm_type);

    /// Compute the overlap of two wave functions acted by complementary operators
    /// Return a map from state to roots of values
    /// Computes the overlap <Ψ(N-1)|Ψ'(N-1)>, where the (N-1)-electron wave function is given by
    /// Ψ(N-1) = h_{pσ} (t) |Ψ (N)> = \sum_{uvw} t_{pw}^{uv} \sum_{σ1} w^+_{σ1} v_{σ1} u_{σ} |Ψ(N)>.
    /// Useful to get the 3-RDM contribution of fully contracted term of two 2-body operators:
    /// \sum_{puvwxyzστθ} v_{pwxy} t_{pzuv} <Ψ(N)| xσ^+ yτ^+ wτ zθ^+ vθ uσ |Ψ(N)>
    std::map<StateInfo, std::vector<double>>
    compute_complementary_H2caa_overlap(ambit::Tensor Tbra, ambit::Tensor Tket,
                                        const std::vector<int>& p_syms, const std::string& name,
                                        bool load = false);

    /// Print a summary of the computation information
    void print_options();

    /// Return a map StateInfo -> size of the determinant space
    std::map<StateInfo, size_t> state_space_size_map() const;

    /// Return a map of StateInfo to the computed nroots of energies
    const std::map<StateInfo, std::vector<double>>& state_energies_map() const;
    /// Return a map of StateInfo to the CI wave functions (deterministic determinant space)
    std::map<StateInfo, std::shared_ptr<psi::Matrix>> state_ci_wfn_map() const;

    /// Pass a set of ActiveSpaceIntegrals to the solver (e.g. an effective Hamiltonian)
    /// @param as_ints the pointer to a set of active-space integrals
    void set_active_space_integrals(std::shared_ptr<ActiveSpaceIntegrals> as_ints);

    /// Pass multipole integrals to the solver (e.g. correlation dressed dipole/quadrupole)
    /// @param as_mp_ints the pointer to a set of multipole integrals
    void set_active_multipole_integrals(std::shared_ptr<ActiveMultipoleIntegrals> as_mp_ints);

    /// Return the map of StateInfo to the wave function file name
    std::map<StateInfo, std::string> state_filename_map() const { return state_filename_map_; }

    /// Save the wave function to disk
    void dump_wave_function();

    /// Set energy convergence
    void set_e_convergence(double e_convergence);

    /// Set residual convergence
    void set_r_convergence(double r_convergence);

    /// Set the maximum number of iterations
    void set_maxiter(int maxiter);

    /// Set if throw an error when Davidson-Liu not converged
    void set_die_if_not_converged(bool value) { die_if_not_converged_ = value; }

    /// Set if read wave function from file as initial guess
    void set_read_initial_guess(bool read_guess) { read_initial_guess_ = read_guess; }

    /// Return the eigen vectors for a given state
    std::vector<ambit::Tensor> eigenvectors(const StateInfo& state) const;
    /// Set unitary matrices for changing orbital basis in RDMs when computing dipole moments
    void set_Uactv(ambit::Tensor& Ua, ambit::Tensor& Ub) {
        Ua_actv_ = Ua;
        Ub_actv_ = Ub;
    }

  protected:
    /// a string that specifies the method used (e.g. "FCI", "ACI", ...)
    std::string solver_type_;

    /// A map of electronic states to the number of roots computed
    ///   {state_1 : n_1, state_2 : n_2, ...}
    /// where state_i specifies the symmetry of a state and n_i is the number of levels computed.
    std::map<StateInfo, size_t> state_nroots_map_;

    /// The information about a previous SCF computation
    std::shared_ptr<SCFInfo> scf_info_;

    /// The MOSpaceInfo object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// User-provided options
    std::shared_ptr<ForteOptions> options_;

    /// The molecular integrals for the active space
    /// This object holds only the integrals for the orbital contained in the
    /// active_mo_vector.
    /// The one-electron integrals and scalar energy contains contributions from the
    /// doubly occupied orbitals specified by the core_mo_ vector.
    std::shared_ptr<ActiveSpaceIntegrals> as_ints_;

    /// The multipole integrals for the active space
    std::shared_ptr<ActiveMultipoleIntegrals> as_mp_ints_;

    /// A map of state symmetries to the associated ActiveSpaceMethod
    std::map<StateInfo, std::shared_ptr<ActiveSpaceMethod>> state_method_map_;

    /// Make sure that the values of <S^2> are consistent with the multiplicity
    void validate_spin(const std::vector<double>& spin2, const StateInfo& state);

    /// Prints a summary of the energies with State info
    void print_energies();

    /// A map of state symmetries to vectors of computed energies under given state symmetry
    std::map<StateInfo, std::vector<double>> state_energies_map_;

    /// A map of state symmetries to vectors of computed average S^2 under given state symmetry
    std::map<StateInfo, std::vector<double>> state_spin2_map_;

    /// A map of state symmetries to the file name of wave function stored on disk
    std::map<StateInfo, std::string> state_filename_map_;

    /// A variable to control printing information
    PrintLevel print_ = PrintLevel::Default;

    /// The energy convergence criterion
    double e_convergence_ = 1.0e-10;

    /// The residual 2-norm convergence criterion
    double r_convergence_ = 1.0e-6;

    /// The maximum number of iterations
    size_t maxiter_ = 100;

    /// Stop if Davidson-Liu not converged
    bool die_if_not_converged_ = true;

    /// Read wave function from disk as initial guess
    bool read_initial_guess_;

    /// Only print the transitions between states with different gas
    bool gas_diff_only_;

    /// Unitary matrices for orbital rotations used to compute dipole moments
    /// The issue is dipole integrals are transformed to semi-canonical orbital basis,
    /// while active-space integrals are in the original orbital basis
    ambit::Tensor Ua_actv_;
    ambit::Tensor Ub_actv_;

    /// Pairs of state info and the contracted CI eigen vectors
    std::map<StateInfo, std::shared_ptr<psi::Matrix>>
        state_contracted_evecs_map_; // TODO move outside?
};                                   // namespace forte

/**
 * @brief Make an active space solver object.
 * @param type a string that specifies the type (e.g. "FCI", "ACI", ...)
 * @param state_nroots_map a map from state symmetry to the number of roots
 * @param scf_info information about a previous SCF computation
 * @param mo_space_info orbital space information
 * @param ints an integral object
 * @param options user-provided options
 * @return a shared pointer for the base class ActiveSpaceMethod
 */
std::shared_ptr<ActiveSpaceSolver> make_active_space_solver(
    const std::string& method, const std::map<StateInfo, size_t>& state_nroots_map,
    std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<MOSpaceInfo> mo_space_info,
    std::shared_ptr<ForteOptions> options,
    std::shared_ptr<ActiveSpaceIntegrals> as_ints = std::shared_ptr<ActiveSpaceIntegrals>());

/**
 * @brief Convert a map of StateInfo to weight lists to a map of StateInfo to number of roots.
 * @param state_weights_map A map of StateInfo to weight lists
 * @return A map of StateInfo to number of states
 */
std::map<StateInfo, size_t>
to_state_nroots_map(const std::map<StateInfo, std::vector<double>>& state_weights_map);

/**
 * @brief Make a list of states and weights.
 * @param options user-provided options
 * @param mo_space_info orbital space information
 * @return a unique pointer to an ActiveSpaceSolver object
 */
std::map<StateInfo, std::vector<double>>
make_state_weights_map(std::shared_ptr<ForteOptions> options,
                       std::shared_ptr<forte::MOSpaceInfo> mo_space_info);

/**
 * @brief Compute the average energy for a set of states
 * @param state_energies_list a map of state -> energies
 * @param state_weight_list a map of state -> weights
 */
double
compute_average_state_energy(const std::map<StateInfo, std::vector<double>>& state_energies_map,
                             const std::map<StateInfo, std::vector<double>>& state_weight_map);

} // namespace forte
