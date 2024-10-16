import pytest

from forte.modules import MoleculeFactory, StateFactory, HF, Workflow


def test_rhf():
    """Test RHF on H2."""

    ref_energy = -1.10015376479352

    # define a molecule
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    """

    # create a data object
    data = MoleculeFactory(xyz).run()
    # create a state object
    data = StateFactory(charge=0, multiplicity=1, sym="ag").run(data)
    # create a HF module
    hf = HF(basis="cc-pVDZ")
    # run the solver
    data = hf.run(data)

    assert data.results.value("hf energy") == pytest.approx(ref_energy, 1.0e-10)


def test_rhf_2():
    """Test RHF on H2."""

    ref_energy = -1.10015376479352

    # define a molecule
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    """

    # create a data object
    seq = Workflow([MoleculeFactory(xyz), StateFactory(charge=0, multiplicity=1, sym="ag"), HF(basis="cc-pVDZ")])
    data = seq.run()

    assert data.results.value("hf energy") == pytest.approx(ref_energy, 1.0e-10)


def test_rhf_3():
    """Test RHF on H2."""

    ref_energy = -1.10015376479352

    # define a molecule
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    """

    with pytest.raises(AssertionError):
        job = Workflow([MoleculeFactory(xyz), HF(basis="cc-pVDZ")])
        data = job.run()

    with pytest.raises(AssertionError):
        job = Workflow([StateFactory(charge=0, multiplicity=1, sym="ag"), HF(basis="cc-pVDZ")])
        data = job.run()


if __name__ == "__main__":
    test_rhf()
    test_rhf_2()
    test_rhf_3()
