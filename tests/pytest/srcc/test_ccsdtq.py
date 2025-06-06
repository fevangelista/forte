import pytest
import forte
import psi4


def test_ccsdtq():
    """Test CCSDTQ on H4 using RHF/DZ orbitals"""

    ref_energy = -2.225370535177  # from psi4

    molecule = psi4.geometry(
        """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    H 0.0 0.0 2.0
    H 0.0 0.0 3.0     
    """
    )

    data = forte.modules.ObjectsUtilPsi4(molecule=molecule, basis="DZ").run()

    cc = forte.modules.GeneralCC(cc_type="cc", max_exc=4)
    data = cc.run(data)
    scf_energy = data.psi_wfn.energy()

    psi4.core.clean()

    energy = data.results.value("energy")

    print(f"  HF energy:     {scf_energy}")
    print(f"  CCSDTQ energy: {energy}")
    print(f"  E - Eref:      {energy - ref_energy}")

    assert energy == pytest.approx(ref_energy, 5.0e-10)


if __name__ == "__main__":
    test_ccsdtq()
