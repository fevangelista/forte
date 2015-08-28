#include <algorithm>
#include <vector>
#include <map>

#include <libmints/molecule.h>

#include "helpers.h"
#include "mrdsrg.h"

namespace psi{ namespace forte{

MRDSRG::MRDSRG(Reference reference,boost::shared_ptr<Wavefunction> wfn,Options &options,ForteIntegrals* ints,std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options,_default_psio_lib_),
      reference_(reference),
      ints_(ints),
      tensor_type_(kCore),
      BTF(new BlockedTensorFactory(options)),
      mo_space_info_(mo_space_info)
{
    // Copy the wavefunction information
//    copy(wfn);

    print_method_banner({"Multireference Driven Similarity Renormalization Group","Chenyang Li"});
    startup();
    print_options();
}

MRDSRG::~MRDSRG(){
    print_comm_time();
}

void MRDSRG::startup()
{
    Eref = reference_.get_Eref();
    BlockedTensor::reset_mo_spaces();

    print_ = options_.get_int("PRINT");

    frozen_core_energy = ints_->frozen_core_energy();

    source_ = options_.get_str("SOURCE");

    s_ = options_.get_double("DSRG_S");
    if(s_ < 0){
        outfile->Printf("\n  S parameter for DSRG must >= 0!");
        throw PSIEXCEPTION("S parameter for DSRG must >= 0!");
    }
    taylor_threshold_ = options_.get_int("TAYLOR_THRESHOLD");
    if(taylor_threshold_ <= 0){
        outfile->Printf("\n  Threshold for Taylor expansion must be an integer greater than 0!");
        throw PSIEXCEPTION("Threshold for Taylor expansion must be an integer greater than 0!");
    }
    taylor_order_ = int(0.5 * (15.0 / taylor_threshold_ + 1)) + 1;

    ntamp_ = options_.get_int("NTAMP");
    intruder_tamp_ = options_.get_double("INTRUDER_TAMP");

    // orbital spaces
    acore_mos = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    bcore_mos = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    aactv_mos = mo_space_info_->get_corr_abs_mo("ACTIVE");
    bactv_mos = mo_space_info_->get_corr_abs_mo("ACTIVE");
    avirt_mos = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");
    bvirt_mos = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");

    // define space labels
    acore_label = "c";
    aactv_label = "a";
    avirt_label = "v";
    bcore_label = "C";
    bactv_label = "A";
    bvirt_label = "V";
    BTF->add_mo_space(acore_label,"mn",acore_mos,AlphaSpin);
    BTF->add_mo_space(bcore_label,"MN",bcore_mos,BetaSpin);
    BTF->add_mo_space(aactv_label,"uvwxyz",aactv_mos,AlphaSpin);
    BTF->add_mo_space(bactv_label,"UVWXYZ",bactv_mos,BetaSpin);
    BTF->add_mo_space(avirt_label,"ef",avirt_mos,AlphaSpin);
    BTF->add_mo_space(bvirt_label,"EF",bvirt_mos,BetaSpin);

    // map space labels to mo spaces
    label_to_spacemo[acore_label[0]] = acore_mos;
    label_to_spacemo[bcore_label[0]] = bcore_mos;
    label_to_spacemo[aactv_label[0]] = aactv_mos;
    label_to_spacemo[bactv_label[0]] = bactv_mos;
    label_to_spacemo[avirt_label[0]] = avirt_mos;
    label_to_spacemo[bvirt_label[0]] = bvirt_mos;

    // define composite spaces
    BTF->add_composite_mo_space("h","ijkl",{acore_label,aactv_label});
    BTF->add_composite_mo_space("H","IJKL",{bcore_label,bactv_label});
    BTF->add_composite_mo_space("p","abcd",{aactv_label,avirt_label});
    BTF->add_composite_mo_space("P","ABCD",{bactv_label,bvirt_label});
    BTF->add_composite_mo_space("g","pqrs",{acore_label,aactv_label,avirt_label});
    BTF->add_composite_mo_space("G","PQRS",{bcore_label,bactv_label,bvirt_label});

    // prepare one-electron integrals
    H = BTF->build(tensor_type_,"H",spin_cases({"gg"}));
    H.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin) value = ints_->oei_a(i[0],i[1]);
        else value = ints_->oei_b(i[0],i[1]);
    });

    // prepare two-electron integrals
    V = BTF->build(tensor_type_,"V",spin_cases({"gggg"}));
    V.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if ((spin[0] == AlphaSpin) && (spin[1] == AlphaSpin)){
            value = ints_->aptei_aa(i[0],i[1],i[2],i[3]);
//            outfile->Printf("\n  [%zu][%zu][%zu][%zu] = %22.15f",i[0],i[1],i[2],i[3],value);
        }
        if ((spin[0] == AlphaSpin) && (spin[1] == BetaSpin))  value = ints_->aptei_ab(i[0],i[1],i[2],i[3]);
        if ((spin[0] == BetaSpin)  && (spin[1] == BetaSpin))  value = ints_->aptei_bb(i[0],i[1],i[2],i[3]);
    });

    // prepare density matrices
    // TODO: future code will store only active Gamma1 and Eta1
    Gamma1 = BTF->build(tensor_type_,"Gamma1",spin_cases({"hh"}));
    Eta1 = BTF->build(tensor_type_,"Eta1",spin_cases({"pp"}));
    (Gamma1.block("cc")).iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;});
    (Gamma1.block("CC")).iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;});
    (Eta1.block("aa")).iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;});
    (Eta1.block("AA")).iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;});
    (Eta1.block("vv")).iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;});
    (Eta1.block("VV")).iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;});
    Gamma1.block("aa")("pq") = reference_.L1a()("pq");
    Gamma1.block("AA")("pq") = reference_.L1b()("pq");
    Eta1.block("aa")("pq") -= reference_.L1a()("pq");
    Eta1.block("AA")("pq") -= reference_.L1b()("pq");

    // prepare two-body density cumulants
    Lambda2 = BTF->build(tensor_type_,"Lambda2",spin_cases({"aaaa"}));
    ambit::Tensor Lambda2_aa = Lambda2.block("aaaa");
    ambit::Tensor Lambda2_aA = Lambda2.block("aAaA");
    ambit::Tensor Lambda2_AA = Lambda2.block("AAAA");
    Lambda2_aa("pqrs") = reference_.L2aa()("pqrs");
    Lambda2_aA("pqrs") = reference_.L2ab()("pqrs");
    Lambda2_AA("pqrs") = reference_.L2bb()("pqrs");

    // prepare three-body density cumulants
    Lambda3 = BTF->build(tensor_type_,"Lambda3",spin_cases({"aaaaaa"}));
    ambit::Tensor Lambda3_aaa = Lambda3.block("aaaaaa");
    ambit::Tensor Lambda3_aaA = Lambda3.block("aaAaaA");
    ambit::Tensor Lambda3_aAA = Lambda3.block("aAAaAA");
    ambit::Tensor Lambda3_AAA = Lambda3.block("AAAAAA");
    Lambda3_aaa("pqrstu") = reference_.L3aaa()("pqrstu");
    Lambda3_aaA("pqrstu") = reference_.L3aab()("pqrstu");
    Lambda3_aAA("pqrstu") = reference_.L3abb()("pqrstu");
    Lambda3_AAA("pqrstu") = reference_.L3bbb()("pqrstu");

    // build Fock matrix (initial guess of one-body Hamiltonian)
    F = BTF->build(tensor_type_,"Fock",spin_cases({"gg"}));
    F["pq"]  = H["pq"];
    F["pq"] += V["pjqi"] * Gamma1["ij"];
    F["pq"] += V["pJqI"] * Gamma1["IJ"];
    F["PQ"]  = H["PQ"];
    F["PQ"] += V["jPiQ"] * Gamma1["ij"];
    F["PQ"] += V["PJQI"] * Gamma1["IJ"];

    // obtain diagonal elements of Fock matrix
    size_t ncmo_ = ints_->ncmo();
    Fa = std::vector<double>(ncmo_);
    Fb = std::vector<double>(ncmo_);
    F.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin and (i[0] == i[1])){
            Fa[i[0]] = value;
        }
        if (spin[0] == BetaSpin and (i[0] == i[1])){
            Fb[i[0]] = value;
        }
    });
}

void MRDSRG::print_options()
{
    // fill in information
    std::vector<std::pair<std::string,int>> calculation_info{
        {"ntamp", ntamp_}};

    std::vector<std::pair<std::string,double>> calculation_info_double{
        {"flow parameter",s_},
        {"taylor expansion threshold",pow(10.0,-double(taylor_threshold_))},
        {"intruder_tamp", intruder_tamp_}};

    std::vector<std::pair<std::string,std::string>> calculation_info_string{
        {"corr_level", options_.get_str("CORR_LEVEL")},
        {"int_type", options_.get_str("INT_TYPE")},
        {"source operator", source_}};

    // print some information
    outfile->Printf("\n\n  ==> Calculation Information <==\n");
    for (auto& str_dim : calculation_info){
        outfile->Printf("\n    %-35s %15d",str_dim.first.c_str(),str_dim.second);
    }
    for (auto& str_dim : calculation_info_double){
        outfile->Printf("\n    %-35s %15.3e",str_dim.first.c_str(),str_dim.second);
    }
    for (auto& str_dim : calculation_info_string){
        outfile->Printf("\n    %-35s %15s",str_dim.first.c_str(),str_dim.second.c_str());
    }
    outfile->Flush();
}

double MRDSRG::renormalized_denominator(double D){
    double Z = std::sqrt(s_) * D;
    if(std::fabs(Z) < std::pow(0.1, taylor_threshold_)){
        return Taylor_Exp(Z, taylor_order_) * std::sqrt(s_);
    }else{
        return (1.0 - std::exp(-s_ * std::pow(D, 2.0))) / D;
    }
}

double MRDSRG::renormalized_denominator_labs(double D){
    double Z = s_ * D;
    if(std::fabs(Z) < std::pow(0.1, taylor_threshold_)){
        return Taylor_Exp_Linear(Z, taylor_order_ * 2) * s_;
    }else{
        return (1.0 - std::exp(-s_ * std::fabs(D))) / D;
    }
}

double MRDSRG::compute_energy(){
    // build initial amplitudes
    outfile->Printf("\n\n  ==> Build Initial Amplitude from DSRG-MRPT2 <==\n");
    T1 = BTF->build(tensor_type_,"T1 Amplitudes",spin_cases({"hp"}));
    T2 = BTF->build(tensor_type_,"T2 Amplitudes",spin_cases({"hhpp"}));
    guess_t2(V,T2);
    guess_t1(F,T2,T1);

    // check initial amplitudes
    analyze_amplitudes("First-Order ",T1,T2);

    // get reference energy
    double Etotal = Eref;

    // compute energy
    switch (corrlevelmap[options_.get_str("CORR_LEVEL")]){
    case LDSRG2:{
        Etotal += compute_energy_ldsrg2();
        break;
    }
    case LDSRG2_P3:{
        break;
    }
    case QDSRG2:{
        break;
    }
    case QDSRG2_P3:{
        break;
    }
    case PT3:{
        Etotal += compute_energy_pt3();
        break;
    }
    default:{
        Etotal += compute_energy_pt2();
    }}

    // transfer integrals if relaxes ref.
    if(options_.get_str("RELAX_REF") == "ONCE"){
        transfer_integrals();
    }

    return Etotal;
}

void MRDSRG::transfer_integrals(){
    // printing
    outfile->Printf("\n\n  ==> De-Normal-Order the DSRG Transformed Hamiltonian <==\n");

    // compute scalar term
    Timer t_scalar;
    std::string str = "Computing the scalar term   ...";
    outfile->Printf("\n    %-35s", str.c_str());
    double scalar0 = Eref + Hbar0 - molecule_->nuclear_repulsion_energy();

    // scalar from Hbar1
    double scalar1 = 0.0;
    Hbar1.block("cc").citerate([&](const std::vector<size_t>& i,const double& value){
        if (i[0] == i[1]) scalar1 -= value;
    });
    Hbar1.block("CC").citerate([&](const std::vector<size_t>& i,const double& value){
        if (i[0] == i[1]) scalar1 -= value;
    });
    scalar1 -= Hbar1["vu"] * Gamma1["uv"];
    scalar1 -= Hbar1["VU"] * Gamma1["UV"];

    // scalar from Hbar2
    double scalar2 = 0.0;
    scalar2 -= 0.25 * Hbar2["xyuv"] * Lambda2["uvxy"];
    scalar2 -= 0.25 * Hbar2["XYUV"] * Lambda2["UVXY"];
    scalar2 -= Hbar2["xYuV"] * Lambda2["uVxY"];
    Hbar2.block("cccc").citerate([&](const std::vector<size_t>& i,const double& value){
        if ((i[0] == i[2]) && (i[1] == i[3])) scalar2 += 0.5 * value;
    });
    Hbar2.block("cCcC").citerate([&](const std::vector<size_t>& i,const double& value){
        if ((i[0] == i[2]) && (i[1] == i[3])) scalar2 += value;
    });
    Hbar2.block("CCCC").citerate([&](const std::vector<size_t>& i,const double& value){
        if ((i[0] == i[2]) && (i[1] == i[3])) scalar2 += 0.5 * value;
    });

    O1 = BTF->build(tensor_type_,"O1",spin_cases({"gg"}));
    O1["pq"] += Hbar2["puqv"] * Gamma1["vu"];
    O1["pq"] += Hbar2["pUqV"] * Gamma1["VU"];
    O1["PQ"] += Hbar2["uPvQ"] * Gamma1["vu"];
    O1["PQ"] += Hbar2["PUQV"] * Gamma1["VU"];
    O1.block("cc").citerate([&](const std::vector<size_t>& i,const double& value){
        if (i[0] == i[1]) scalar2 += value;
    });
    O1.block("CC").citerate([&](const std::vector<size_t>& i,const double& value){
        if (i[0] == i[1]) scalar2 += value;
    });
    scalar2 += 0.5 * Gamma1["uv"] * Hbar2["vyux"] * Gamma1["xy"];
    scalar2 += 0.5 * Gamma1["UV"] * Hbar2["VYUX"] * Gamma1["XY"];
    scalar2 += Gamma1["uv"] * Hbar2["vYuX"] * Gamma1["XY"];

    double scalar = scalar0 + scalar1 + scalar2;
    outfile->Printf("  Done. Timing %10.3f s", t_scalar.get());

    // compute one-body term
    Timer t_one;
    str = "Computing the one-body term ...";
    outfile->Printf("\n    %-35s", str.c_str());
    O1.scale(-1.0);
    O1["pq"] += Hbar1["pq"];
    O1["PQ"] += Hbar1["PQ"];
    BlockedTensor temp = BTF->build(tensor_type_,"temp",spin_cases({"cc"}));
    temp.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (i[0] == i[1]) value = 1.0;
    });
    O1["pq"] -= Hbar2["pmqn"] * temp["nm"];
    O1["pq"] -= Hbar2["pMqN"] * temp["NM"];
    O1["PQ"] -= Hbar2["mPnQ"] * temp["nm"];
    O1["PQ"] -= Hbar2["PMQN"] * temp["NM"];
    outfile->Printf("  Done. Timing %10.3f s", t_one.get());

    // update integrals
    Timer t_int;
    str = "Updating integrals          ...";
    outfile->Printf("\n    %-35s", str.c_str());
    ints_->set_scalar(scalar);
    O1.citerate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,const double& value){
        if (spin[0] == AlphaSpin){
            ints_->set_oei(i[0],i[1],value,true);
        }else{
            ints_->set_oei(i[0],i[1],value,false);
        }
    });
    Hbar2.citerate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,const double& value){
        if ((spin[0] == AlphaSpin) && (spin[1] == AlphaSpin)){
            ints_->set_tei(i[0],i[1],i[2],i[3],value,true,true);
//            outfile->Printf("\n  [%zu][%zu][%zu][%zu] = %22.15f",i[0],i[1],i[2],i[3],value);
        }else if ((spin[0] == AlphaSpin) && (spin[1] == BetaSpin)){
            ints_->set_tei(i[0],i[1],i[2],i[3],value,true,false);
        }else if ((spin[0] == BetaSpin)  && (spin[1] == BetaSpin)){
            ints_->set_tei(i[0],i[1],i[2],i[3],value,false,false);
        }
    });
    outfile->Printf("  Done. Timing %10.3f s", t_int.get());

    // print scalar
    outfile->Printf("\n\n  ==> Scalar of the Electronic Hamiltonian (wrt True Vacuum) <==\n");
    outfile->Printf("\n    %-30s = %22.15f", "Scalar0", scalar0);
    outfile->Printf("\n    %-30s = %22.15f", "Scalar1", scalar1);
    outfile->Printf("\n    %-30s = %22.15f", "Scalar2", scalar2);
    outfile->Printf("\n    %-30s = %22.15f", "Total", scalar);

    ints_->update_integrals(false);
}

}}