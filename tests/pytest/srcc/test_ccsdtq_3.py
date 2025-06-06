import pytest
import forte
import psi4


@pytest.mark.skip(reason="This is a long test")
def test_ccsdtq_3():
    """Test CCSDTQ on Ne using RHF/cc-pVDZ orbitals"""

    ref_energy = -128.679014931  # from Evangelista, J. Chem. Phys. 134, 224102 (2011).

    molecule = psi4.geometry("Ne 0.0 0.0 0.0")

    data = forte.modules.ObjectsUtilPsi4(
        molecule=molecule, basis="cc-pVDZ", mo_spaces={"FROZEN_DOCC": [1, 0, 0, 0, 0, 0, 0, 0]}
    ).run()

    scf_energy = data.psi_wfn.energy()
    cc = forte.modules.GeneralCC(cc_type="cc", max_exc=4, e_convergence=1.0e-10)
    data = cc.run(data)

    psi4.core.clean()

    energy = data.results.value("energy")

    print(energy - ref_energy)
    assert energy == pytest.approx(ref_energy, 1.0e-9)


if __name__ == "__main__":
    test_ccsdtq_3()
