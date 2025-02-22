/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2025 by its authors (see COPYING, COPYING.LESSER,
 * AUTHORS).
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

#include <algorithm>
#include <format>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/wavefunction.h"

#include "helpers/string_algorithms.h"

#include "base_classes/mo_space_info.h"

using namespace psi;

namespace forte {

const std::vector<std::string> MOSpaceInfo::elementary_spaces_{
    "FROZEN_DOCC", "RESTRICTED_DOCC", "GAS1",       "GAS2", "GAS3", "GAS4", "GAS5",
    "GAS6",        "RESTRICTED_UOCC", "FROZEN_UOCC"};

const std::vector<std::string> MOSpaceInfo::composite_spaces_{
    {"ALL", "FROZEN", "CORRELATED", "ACTIVE", "INACTIVE_DOCC", "INACTIVE_UOCC", "GENERALIZED HOLE",
     "GENERALIZED PARTICLE", "CORE", "VIRTUAL"}};

/// Defines composite orbital spaces
const std::map<std::string, std::vector<std::string>> MOSpaceInfo::composite_spaces_def_{
    {"ALL",
     {"FROZEN_DOCC", "RESTRICTED_DOCC", "GAS1", "GAS2", "GAS3", "GAS4", "GAS5", "GAS6",
      "RESTRICTED_UOCC", "FROZEN_UOCC"}},
    {"FROZEN", {"FROZEN_DOCC", "FROZEN_UOCC"}},
    {"CORRELATED",
     {"RESTRICTED_DOCC", "GAS1", "GAS2", "GAS3", "GAS4", "GAS5", "GAS6", "RESTRICTED_UOCC"}},
    {"ACTIVE", {"GAS1", "GAS2", "GAS3", "GAS4", "GAS5", "GAS6"}},
    {"INACTIVE_DOCC", {"FROZEN_DOCC", "RESTRICTED_DOCC"}},
    {"INACTIVE_UOCC", {"RESTRICTED_UOCC", "FROZEN_UOCC"}},
    // Spaces for multireference calculations
    {"GENERALIZED HOLE", {"RESTRICTED_DOCC", "GAS1", "GAS2", "GAS3", "GAS4", "GAS5", "GAS6"}},
    {"GENERALIZED PARTICLE", {"GAS1", "GAS2", "GAS3", "GAS4", "GAS5", "GAS6", "RESTRICTED_UOCC"}},
    {"CORE", {"RESTRICTED_DOCC"}},
    {"VIRTUAL", {"RESTRICTED_UOCC"}},
    {"FROZEN_DOCC", {"FROZEN_DOCC"}},
    {"FROZEN_UOCC", {"FROZEN_UOCC"}},
    {"RESTRICTED_DOCC", {"RESTRICTED_DOCC"}},
    {"RESTRICTED_UOCC", {"RESTRICTED_UOCC"}},
    {"GAS1", {"GAS1"}},
    {"GAS2", {"GAS2"}},
    {"GAS3", {"GAS3"}},
    {"GAS4", {"GAS4"}},
    {"GAS5", {"GAS5"}},
    {"GAS6", {"GAS6"}}};

const std::vector<std::string> MOSpaceInfo::elementary_spaces_priority_{"GAS1",
                                                                        "RESTRICTED_UOCC",
                                                                        "RESTRICTED_DOCC",
                                                                        "FROZEN_DOCC",
                                                                        "FROZEN_UOCC",
                                                                        "GAS2",
                                                                        "GAS3",
                                                                        "GAS4",
                                                                        "GAS5",
                                                                        "GAS6"};

MOSpaceInfo::MOSpaceInfo(const psi::Dimension& nmopi, const std::string& point_group)
    : symmetry_(point_group), nirrep_(nmopi.n()), nmopi_(nmopi) {}

std::string MOSpaceInfo::str() const {
    std::string s;
    for (const auto& space : elementary_spaces()) {
        s += "space: " + space + " ";
        const auto dim = dimension(space);
        for (size_t h = 0, maxh = static_cast<size_t>(dim.n()); h < maxh; h++) {
            s += " " + std::to_string(dim[h]);
        }
        s += '\n';
    }
    return s;
}

size_t MOSpaceInfo::nirrep() const { return nirrep_; }

const std::vector<std::string>& MOSpaceInfo::irrep_labels() const {
    return symmetry_.irrep_labels();
}

const std::string& MOSpaceInfo::irrep_label(size_t h) const { return symmetry_.irrep_label(h); }

std::string MOSpaceInfo::point_group_label() const { return symmetry_.point_group_label(); }

const std::vector<std::string>& MOSpaceInfo::elementary_spaces() {
    return MOSpaceInfo::elementary_spaces_;
}

const std::vector<std::string>& MOSpaceInfo::composite_spaces() {
    return MOSpaceInfo::composite_spaces_;
}

const std::map<std::string, std::vector<std::string>>& MOSpaceInfo::composite_spaces_def() {
    return MOSpaceInfo::composite_spaces_def_;
}

size_t MOSpaceInfo::size(const std::string& space) const {
    size_t s = 0;
    if (composite_spaces_def().count(space) == 0) {
        std::string msg = "\n  MOSpaceInfo::size - composite space " + space + " is not defined.";
        throw std::runtime_error(msg);
    } else {
        for (const auto& el_space : composite_spaces_def().at(space)) {
            if (mo_spaces_.count(el_space))
                s += mo_spaces_.at(el_space).first.sum();
        }
    }
    return s;
}

psi::Dimension MOSpaceInfo::dimension(const std::string& space) const {
    psi::Dimension result(nirrep_);
    if (composite_spaces_def().count(space) == 0) {
        std::string msg =
            "\n  MOSpaceInfo::dimension - composite space " + space + " is not defined.";
        throw std::runtime_error(msg);
    } else {
        for (const auto& el_space : composite_spaces_def().at(space)) {
            if (mo_spaces_.count(el_space))
                result += mo_spaces_.at(el_space).first;
        }
    }
    return result;
}

std::vector<int> MOSpaceInfo::symmetry(const std::string& space) const {
    psi::Dimension dims = dimension(space);
    std::vector<int> result;
    for (size_t h = 0; h < dims.n(); ++h) {
        fill_n(back_inserter(result), dims[h], h); // insert h for dims[h] times
    }
    return result;
}

std::vector<size_t> MOSpaceInfo::absolute_mo(const std::string& space) const {
    std::vector<size_t> result;
    if (composite_spaces_def().count(space) == 0) {
        std::string msg =
            "\n  MOSpaceInfo::absolute_mo - composite space " + space + " is not defined.";
        throw std::runtime_error(msg);
        ;
    } else {
        std::vector<std::vector<size_t>> mo_list(nirrep_);
        for (const auto& el_space : composite_spaces_def().at(space)) {
            if (mo_spaces_.count(el_space)) {
                auto& vec_mo_info = mo_spaces_.at(el_space).second;
                for (auto& mo_info : vec_mo_info) {
                    size_t h = std::get<1>(mo_info);            // <- the orbital irrep
                    mo_list[h].push_back(std::get<0>(mo_info)); // <- grab the absolute index
                }
            }
        }
        for (const auto& irrep : mo_list) {
            for (const auto& p : irrep) {
                result.push_back(p);
            }
        }
    }
    return result;
}

std::vector<size_t> MOSpaceInfo::corr_absolute_mo(const std::string& space) const {
    std::vector<size_t> result;
    if (composite_spaces_def().count(space) == 0) {
        std::string msg =
            "\n  MOSpaceInfo::corr_absolute_mo - composite space " + space + " is not defined.";
        throw std::runtime_error(msg);
        ;
    } else {
        std::vector<std::vector<size_t>> mo_list(nirrep_);
        // Loop over all the spaces
        for (const auto& el_space : composite_spaces_def().at(space)) {
            if (mo_spaces_.count(el_space)) {
                auto& vec_mo_info = mo_spaces_.at(el_space).second;
                for (auto& [p, h, _] : vec_mo_info) {
                    mo_list[h].push_back(mo_to_cmo_[p]); // <- grab the absolute index
                                                         // and convert to correlated MOs
                }
            }
        }
        for (const auto& irrep : mo_list) {
            for (const auto& p : irrep) {
                result.push_back(p);
            }
        }
    }
    return result;
}

std::vector<std::pair<size_t, size_t>> MOSpaceInfo::relative_mo(const std::string& space) const {
    std::vector<std::pair<size_t, size_t>> result;
    if (composite_spaces_def().count(space) == 0) {
        std::string msg =
            "\n  MOSpaceInfo::relative_mo - composite space " + space + " is not defined.";
        throw std::runtime_error(msg);
        ;
    } else {
        std::vector<std::vector<std::pair<size_t, size_t>>> mo_list(nirrep_);

        for (const auto& el_space : composite_spaces_def().at(space)) {
            if (mo_spaces_.count(el_space)) {
                auto& vec_mo_info = mo_spaces_.at(el_space).second;
                for (auto& mo_info : vec_mo_info) {
                    size_t h = std::get<1>(mo_info); // <- the orbital irrep
                    mo_list[h].push_back(std::make_pair(
                        std::get<1>(mo_info),
                        std::get<2>(mo_info))); // <- grab the irrep and relative index
                }
            }
        }
        for (const auto& irrep : mo_list) {
            for (const auto& p : irrep) {
                result.push_back(p);
            }
        }
    }
    return result;
}

bool MOSpaceInfo::contained_in_space(const std::string& space,
                                     const std::string& composite_space) const {
    if (composite_spaces_def().count(space) * composite_spaces_def().count(composite_space) == 0) {
        std::string msg = "\n  MOSpaceInfo::contained_in_space - space " + space +
                          " or composite space " + composite_space + " is not defined.";
        throw std::runtime_error(msg);
    }

    std::unordered_set<std::string> composite_spaces(
        composite_spaces_def().at(composite_space).begin(),
        composite_spaces_def().at(composite_space).end());
    for (const std::string& s : composite_spaces_def().at(space)) {
        if (composite_spaces.find(s) == composite_spaces.end()) {
            return false;
        }
    }
    return true;
}

std::vector<size_t> MOSpaceInfo::pos_in_space(const std::string& space,
                                              const std::string& composite_space) {
    // make sure that space is contained in composite_space
    if (not contained_in_space(space, composite_space)) {
        std::string msg = "\n  MOSpaceInfo::pos_in_space - space " + space +
                          " is not contained in composite space " + composite_space + " .";
        throw std::runtime_error(msg);
    }

    std::vector<size_t> result;
    if (composite_spaces_def().count(space) * composite_spaces_def().count(composite_space) == 0) {
        std::string msg = "\n  MOSpaceInfo::pos_in_space - space " + space +
                          " or composite space " + composite_space + " is not defined.";
        throw std::runtime_error(msg);
        ;
    }

    // make sure that space is contained in composite_space
    for (auto s : composite_spaces_def().at(space)) {
        auto it = find(composite_spaces_def().at(composite_space).begin(),
                       composite_spaces_def().at(composite_space).end(), s);
        if (it == composite_spaces_def().at(composite_space).end()) {
            std::string msg = "\n  MOSpaceInfo::pos_in_space - space " + s +
                              " is not contained in composite space " + composite_space + " .";
            throw std::runtime_error(msg);
            ;
        }
    }

    auto abs_space = absolute_mo(space);
    auto abs_composite_space = absolute_mo(composite_space);
    std::unordered_map<size_t, size_t> composite_space_hash;
    size_t k = 0;
    for (size_t p : abs_composite_space) {
        composite_space_hash[p] = k;
        k += 1;
    }
    for (size_t p : abs_space) {
        result.push_back(composite_space_hash[p]);
    }
    return result;
}

psi::Slice MOSpaceInfo::range(const std::string& space) {
    if (composite_spaces_def().count(space) == 0) {
        std::string msg = "\n  MOSpaceInfo::range - composite space " + space + " is not defined.";
        throw std::runtime_error(msg);
        ;
    }

    // make sure the elementary spaces in composite space are consecutive
    int start = std::find(elementary_spaces_.begin(), elementary_spaces_.end(),
                          composite_spaces_def().at(space)[0]) -
                elementary_spaces_.begin();

    for (int i = 1, size = composite_spaces_def().at(space).size(); i < size; ++i) {
        if (composite_spaces_def().at(space)[i] != elementary_spaces_[start + i]) {
            std::string msg = "\n  MOSpaceInfo::range - elementary spaces in composite space " +
                              space + " are not consecutive.";
            throw std::runtime_error(msg);
            ;
        }
    }

    // prepare start and end psi::Dimension
    psi::Dimension dim_start(nirrep_, space + " begin");
    for (int i = 0; i < start; ++i) {
        dim_start += mo_spaces_[elementary_spaces_[i]].first;
    }

    psi::Dimension dim_end(nirrep_, space + " end");
    dim_end += dim_start;
    for (int i = start, end = start + composite_spaces_def().at(space).size(); i < end; ++i) {
        dim_end += mo_spaces_[elementary_spaces_[i]].first;
    }

    return psi::Slice(dim_start, dim_end);
}

void MOSpaceInfo::read_from_map(const std::map<std::string, std::vector<size_t>>& mo_space_map) {
    // Capitalize the space names to allow the user to use both lower or upper case strings
    std::map<std::string, std::vector<size_t>> mo_space_map_capitalized;
    for (const auto& el : mo_space_map) {
        mo_space_map_capitalized[upper_string(el.first)] = el.second;
    }
    // Read the elementary spaces
    for (const std::string& space : elementary_spaces_) {
        std::pair<SpaceInfo, bool> result = read_mo_space_from_map(space, mo_space_map_capitalized);
        if (result.second) {
            mo_spaces_[space] = result.first;
        }
    }
    // Treat "ACTIVE" in a special way
    std::pair<SpaceInfo, bool> result = read_mo_space_from_map("ACTIVE", mo_space_map_capitalized);
    if (result.second) {
        mo_spaces_["GAS1"] = result.first;
    }
}

void MOSpaceInfo::compute_space_info() {
    outfile->Printf("\n\n  ==> MO Space Information <==\n");
    // Handle frozen core

    // Count the assigned orbitals
    psi::Dimension unassigned = nmopi_;
    for (auto& str_si : mo_spaces_) {
        unassigned -= str_si.second.first;
    }

    for (size_t h = 0; h < nirrep_; ++h) {
        if (unassigned[h] < 0) {
            // Throw and exception if there are more orbitals assigned to an irrep than there
            // are available
            auto msg = std::format("There is an error in the definition of the orbital spaces. "
                                   " Total assigned MOs "
                                   "for irrep {} is {} leaving {} orbitals unassigned.",
                                   h, nmopi_[h], unassigned[h]);
            outfile->Printf("\n%s", msg.c_str());
            throw std::runtime_error(msg);
        }
    }

    // Adjust size of undefined spaces
    for (std::string space : elementary_spaces_priority_) {
        // Assign MOs to the undefined space with the highest priority
        if (not mo_spaces_.count(space)) {
            std::vector<MOInfo> vec_mo_info;
            mo_spaces_[space] = std::make_pair(unassigned, vec_mo_info);
            for (size_t h = 0; h < nirrep_; ++h) {
                unassigned[h] = 0;
            }
        }
    }
    if (unassigned.sum() != 0) {
        outfile->Printf("\n  There is an error in the definition of the "
                        "orbital spaces.  There are %d unassigned MOs.",
                        unassigned.sum());
        exit(1);
    }

    // Compute orbital mappings
    for (size_t h = 0, p_abs = 0; h < nirrep_; ++h) {
        size_t p_rel = 0;
        for (std::string space : elementary_spaces_) {
            size_t n = mo_spaces_[space].first[h];
            for (size_t q = 0; q < n; ++q) {
                mo_spaces_[space].second.push_back(std::make_tuple(p_abs, h, p_rel));
                p_abs += 1;
                p_rel += 1;
            }
        }
    }

    // Compute the MO to correlated MO mapping
    std::vector<size_t> vec(nmopi_.sum());
    std::iota(vec.begin(), vec.end(), 0);

    // Remove the frozen core/virtuals
    for (MOInfo& mo_info : mo_spaces_["FROZEN_DOCC"].second) {
        size_t removed_mo = std::get<0>(mo_info);
        vec.erase(std::remove(vec.begin(), vec.end(), removed_mo), vec.end());
    }
    for (MOInfo& mo_info : mo_spaces_["FROZEN_UOCC"].second) {
        size_t removed_mo = std::get<0>(mo_info);
        vec.erase(std::remove(vec.begin(), vec.end(), removed_mo), vec.end());
    }

    mo_to_cmo_.assign(nmopi_.sum(), 1000000000);
    for (size_t n = 0; n < vec.size(); ++n) {
        mo_to_cmo_[vec[n]] = n;
    }

    // Define composite spaces

    // Print the space information
    size_t label_size = 1;
    for (std::string space : elementary_spaces_) {
        label_size = std::max(space.size(), label_size);
    }

    int banner_width = label_size + 4 + 6 * (nirrep_ + 1);

    outfile->Printf("\n  %s", std::string(banner_width, '-').c_str());
    outfile->Printf("\n    %s", std::string(label_size, ' ').c_str());
    for (size_t h = 0; h < nirrep_; ++h)
        outfile->Printf(" %5s", irrep_label(h).c_str());
    outfile->Printf("   Sum");
    outfile->Printf("\n  %s", std::string(banner_width, '-').c_str());

    for (std::string space : elementary_spaces_) {
        psi::Dimension& dim = mo_spaces_[space].first;
        outfile->Printf("\n    %-*s", label_size, space.c_str());
        for (size_t h = 0; h < nirrep_; ++h) {
            outfile->Printf("%6d", dim[h]);
        }
        outfile->Printf("%6d", dim.sum());
    }
    outfile->Printf("\n    %-*s", label_size, "Total");
    for (size_t h = 0; h < nirrep_; ++h) {
        outfile->Printf("%6d", nmopi_[h]);
    }
    outfile->Printf("%6d", nmopi_.sum());
    outfile->Printf("\n  %s", std::string(banner_width, '-').c_str());
}

// std::pair<SpaceInfo, bool> MOSpaceInfo::read_mo_space(const std::string& space,
//                                                       std::shared_ptr<ForteOptions> options)
//                                                       {
//     bool read = false;
//     psi::Dimension space_dim(nirrep_);
//     std::vector<MOInfo> vec_mo_info;
//     if (not options->exists(space)) {
//         SpaceInfo space_info(space_dim, vec_mo_info);
//         return std::make_pair(space_info, false);
//     }
//     size_t vec_size = options->get_int_list(space).size();
//     if (vec_size == nirrep_) {
//         for (size_t h = 0; h < nirrep_; ++h) {
//             space_dim[h] = options->get_int_list(space)[h];
//         }
//         read = true;
//         outfile->Printf("\n  Read options for space %s", space.c_str());
//     } else if (vec_size > 0) {
//         std::string msg = "\n  The size of space " + space + " (" + std::to_string(vec_size)
//         +
//                           ") does not match the number of irreducible representations (" +
//                           std::to_string(nirrep_) + ").";
//         outfile->Printf("\n%s", msg.c_str());
//         throw std::runtime_error(msg);
//     }
//     SpaceInfo space_info(space_dim, vec_mo_info);
//     return std::make_pair(space_info, read);
// }

std::vector<std::string> MOSpaceInfo::nonzero_gas_spaces() const {
    std::vector<std::string> nonzero_gas;

    auto gas_spaces = composite_spaces_def().at("ACTIVE");
    for (const std::string& gas_name : gas_spaces) {
        if (size(gas_name) != 0) {
            nonzero_gas.push_back(gas_name);
        }
    }

    return nonzero_gas;
}

std::pair<SpaceInfo, bool> MOSpaceInfo::read_mo_space_from_map(
    const std::string& space, const std::map<std::string, std::vector<size_t>>& mo_space_map) {
    bool read = false;
    psi::Dimension space_dim(nirrep_);
    std::vector<MOInfo> vec_mo_info;

    // lookup the space
    auto it = mo_space_map.find(space);
    if (it != mo_space_map.end()) {
        const auto& dim = mo_space_map.at(space);
        if (dim.size() == nirrep_) {
            for (size_t h = 0; h < nirrep_; ++h) {
                space_dim[h] = dim[h];
            }
            read = true;
        } else {
            throw std::runtime_error("\n  The size of space vector does not match the number of "
                                     "irreducible representations.");
        }
    }
    SpaceInfo space_info(space_dim, vec_mo_info);
    return std::make_pair(space_info, read);
}

std::shared_ptr<MOSpaceInfo>
make_mo_space_info_from_map(const psi::Dimension& nmopi, const std::string& point_group,
                            const std::map<std::string, std::vector<size_t>>& mo_space_map) {

    auto mo_space_info = std::make_shared<MOSpaceInfo>(nmopi, point_group);
    mo_space_info->read_from_map(mo_space_map);
    mo_space_info->compute_space_info();
    return mo_space_info;
}

} // namespace forte
