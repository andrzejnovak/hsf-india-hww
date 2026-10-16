"""The event features of the classifier, shared by train.py (offline) and fit.py (run time). numpy only.

The kit's ten (examples/nn_fit) and twelve more: the charge product, each lepton's transverse mass with the missing
ET, the angles between the missing ET and the leptons and the dilepton system, the projected missing ET of the
WW analyses, both leptons' |eta|, and the two jets with their mass and separation (the VBF signal).
"""
import numpy as np

import hww as H

FEATURES = ["lep_pt_0", "lep_pt_1", "met_et", "dphi", "mll", "ptll", "mt", "jet_n", "deta", "same_flavour",
            "charge_prod", "mt_l0", "mt_l1", "dphi_met_lmin", "dphi_ll_met", "met_proj", "abs_eta_0", "abs_eta_1",
            "jet_pt_0", "jet_pt_1", "mjj", "deta_jj"]


def _dphi(a, b):
    return np.abs(np.mod(a - b + 3.0 * np.pi, 2.0 * np.pi) - np.pi)


def features(ev):
    """The feature matrix of a set of events, in the order of FEATURES."""
    f = lambda c: np.asarray(ev[c], float)
    k = H.kinematics(ev)
    pt0, pt1, phi0, phi1, eta0, eta1 = f("lep_pt_0"), f("lep_pt_1"), f("lep_phi_0"), f("lep_phi_1"), f("lep_eta_0"), f("lep_eta_1")
    met, mphi = f("met_et"), f("met_phi")
    d0, d1 = _dphi(mphi, phi0), _dphi(mphi, phi1)
    dmin = np.minimum(d0, d1)
    phill = np.arctan2(pt0 * np.sin(phi0) + pt1 * np.sin(phi1), pt0 * np.cos(phi0) + pt1 * np.cos(phi1))
    j0, j1 = f("jet_pt_0"), f("jet_pt_1")
    two = (j0 > 0) & (j1 > 0)
    e0, e1 = j0 * np.cosh(f("jet_eta_0")), j1 * np.cosh(f("jet_eta_1"))
    px = j0 * np.cos(f("jet_phi_0")) + j1 * np.cos(f("jet_phi_1"))
    py = j0 * np.sin(f("jet_phi_0")) + j1 * np.sin(f("jet_phi_1"))
    pz = j0 * np.sinh(f("jet_eta_0")) + j1 * np.sinh(f("jet_eta_1"))
    mjj = np.where(two, np.sqrt(np.clip((e0 + e1) ** 2 - px ** 2 - py ** 2 - pz ** 2, 0.0, None)), 0.0)
    cols = {"lep_pt_0": pt0, "lep_pt_1": pt1, "met_et": met, "dphi": k["dphi"], "mll": k["mll"], "ptll": k["ptll"],
            "mt": k["mt"], "jet_n": f("jet_n"), "deta": np.abs(eta0 - eta1),
            "same_flavour": (ev["lep_type_0"] == ev["lep_type_1"]).astype(float),
            "charge_prod": f("lep_charge_0") * f("lep_charge_1"),
            "mt_l0": np.sqrt(np.clip(2.0 * pt0 * met * (1.0 - np.cos(d0)), 0.0, None)),
            "mt_l1": np.sqrt(np.clip(2.0 * pt1 * met * (1.0 - np.cos(d1)), 0.0, None)),
            "dphi_met_lmin": dmin, "dphi_ll_met": _dphi(phill, mphi),
            "met_proj": np.where(dmin < np.pi / 2, met * np.sin(dmin), met),
            "abs_eta_0": np.abs(eta0), "abs_eta_1": np.abs(eta1), "jet_pt_0": j0, "jet_pt_1": j1,
            "mjj": mjj, "deta_jj": np.where(two, np.abs(f("jet_eta_0") - f("jet_eta_1")), 0.0)}
    return np.column_stack([cols[c] for c in FEATURES])
