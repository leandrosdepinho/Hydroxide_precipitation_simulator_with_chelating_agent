# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from scipy.signal import savgol_filter


# =============================================================================
# DATABASE
# =============================================================================

DATABASE = {
    "Hydroxide": {
        "pkas": [14.0],
        "charge_anion": 1,

        "Metals": {
            "Ag+": {"ksp": 2.0e-8, "x": 1, "y": 1},
            "Al3+": {"ksp": 1.0e-33, "x": 1, "y": 3},
            "Ba2+": {"ksp": 5.0e-3, "x": 1, "y": 2},
            "Be2+": {"ksp": 6.9e-22, "x": 1, "y": 2},
            "Bi3+": {"ksp": 4.0e-31, "x": 1, "y": 3},
            "Ca2+": {"ksp": 5.5e-6, "x": 1, "y": 2},
            "Cd2+": {"ksp": 2.5e-14, "x": 1, "y": 2},
            "Ce3+": {"ksp": 1.6e-20, "x": 1, "y": 3},
            "Ce4+": {"ksp": 1.5e-51, "x": 1, "y": 4},
            "Co2+": {"ksp": 1.3e-15, "x": 1, "y": 2},
            "Cr3+": {"ksp": 6.3e-31, "x": 1, "y": 3},
            "Cu2+": {"ksp": 2.2e-20, "x": 1, "y": 2},
            "Dy3+": {"ksp": 1.4e-22, "x": 1, "y": 3},
            "Er3+": {"ksp": 1.3e-22, "x": 1, "y": 3},
            "Eu3+": {"ksp": 3.4e-24, "x": 1, "y": 3},
            "Fe2+": {"ksp": 8.0e-16, "x": 1, "y": 2},
            "Fe3+": {"ksp": 1.0e-39, "x": 1, "y": 3},
            "Ga3+": {"ksp": 7.1e-36, "x": 1, "y": 3},
            "Gd3+": {"ksp": 1.8e-23, "x": 1, "y": 3},
            "Hf4+": {"ksp": 1.5e-52, "x": 1, "y": 4},
            "Hg2+": {"ksp": 3.0e-26, "x": 1, "y": 2},
            "Ho3+": {"ksp": 1.5e-22, "x": 1, "y": 3},
            "In3+": {"ksp": 1.0e-33, "x": 1, "y": 3},
            "La3+": {"ksp": 2.0e-21, "x": 1, "y": 3},
            "Lu3+": {"ksp": 2.5e-24, "x": 1, "y": 3},
            "Mg2+": {"ksp": 1.5e-11, "x": 1, "y": 2},
            "Mn2+": {"ksp": 1.9e-13, "x": 1, "y": 2},
            "Nb5+": {"ksp": 1.0e-62, "x": 1, "y": 5},
            "Nd3+": {"ksp": 1.0e-31, "x": 1, "y": 3},
            "Ni2+": {"ksp": 2.0e-15, "x": 1, "y": 2},
            "Pb2+": {"ksp": 1.2e-15, "x": 1, "y": 2},
            "Pr3+": {"ksp": 1.0e-23, "x": 1, "y": 3},
            "Sc3+": {"ksp": 8.0e-31, "x": 1, "y": 3},
            "Sm3+": {"ksp": 9.0e-25, "x": 1, "y": 3},
            "Sn2+": {"ksp": 5.0e-26, "x": 1, "y": 2},
            "Sn4+": {"ksp": 1.0e-56, "x": 1, "y": 4},
            "Sr2+": {"ksp": 3.2e-4, "x": 1, "y": 2},
            "Ta5+": {"ksp": 1.0e-64, "x": 1, "y": 5},
            "Tb3+": {"ksp": 1.6e-24, "x": 1, "y": 3},
            "Th4+": {"ksp": 1.0e-50, "x": 1, "y": 4},
            "Ti4+": {"ksp": 1.0e-53, "x": 1, "y": 4},
            "Tl+": {"ksp": 1.4e-4, "x": 1, "y": 1},
            "Tl3+": {"ksp": 1.7e-39, "x": 1, "y": 3},
            "Tm3+": {"ksp": 1.1e-22, "x": 1, "y": 3},
            "U4+": {"ksp": 1.0e-52, "x": 1, "y": 4},
            "UO2_2+": {"ksp": 1.1e-22, "x": 1, "y": 2},
            "V3+": {"ksp": 7.0e-36, "x": 1, "y": 3},
            "VO_2+": {"ksp": 7.9e-24, "x": 1, "y": 2},
            "Y3+": {"ksp": 8.0e-23, "x": 1, "y": 3},
            "Yb3+": {"ksp": 2.8e-24, "x": 1, "y": 3},
            "Zn2+": {"ksp": 3.0e-17, "x": 1, "y": 2},
            "Zr4+": {"ksp": 1.0e-62, "x": 1, "y": 4},
        }
    },

    "Chelators": {

        # ---------------------------------------------------------------------
        # EDTA
        # ---------------------------------------------------------------------
        "EDTA": {
            "pkas": [0.0, 1.5, 2.0, 2.66, 6.16, 10.24],

            "log_betas": {
                "Ag+": [7.3],
                "Al3+": [16.1],
                "Ba2+": [7.9],
                "Be2+": [9.2],
                "Bi3+": [27.8],
                "Ca2+": [10.6],
                "Cd2+": [16.5],
                "Ce3+": [16.0],
                "Ce4+": [24.4],
                "Co2+": [16.3],
                "Cr3+": [23.4],
                "Cu2+": [18.8],
                "Dy3+": [18.3],
                "Er3+": [18.9],
                "Eu3+": [17.3],
                "Fe2+": [14.3],
                "Fe3+": [25.1],
                "Ga3+": [20.3],
                "Gd3+": [17.4],
                "Hf4+": [29.5],
                "Hg2_2+": [0.0],
                "Hg2+": [21.8],
                "Ho3+": [18.6],
                "In3+": [25.0],
                "La3+": [15.5],
                "Lu3+": [19.8],
                "Mg2+": [8.7],
                "Mn2+": [13.8],
                "Nb5+": [0.0],
                "Nd3+": [16.6],
                "Ni2+": [18.6],
                "Pb2+": [18.0],
                "Pr3+": [16.4],
                "Sc3+": [23.1],
                "Sm3+": [17.1],
                "Sn2+": [22.1],
                "Sn4+": [0.0],
                "Sr2+": [8.7],
                "Ta5+": [0.0],
                "Tb3+": [17.9],
                "Th4+": [23.2],
                "Ti4+": [17.3],
                "Tl+": [2.2],
                "Tl3+": [22.5],
                "Tm3+": [19.3],
                "U4+": [25.8],
                "UO2_2+": [10.2],
                "V3+": [26.0],
                "VO_2+": [18.8],
                "Y3+": [18.1],
                "Yb3+": [19.5],
                "Zn2+": [16.5],
                "Zr4+": [29.5],
            }
        },

        # ---------------------------------------------------------------------
        # Glycine
        # ---------------------------------------------------------------------
        "Glycine": {
            "pkas": [2.34, 9.60],

            "log_betas": {
                "Ag+": [3.4, 6.9],
                "Al3+": [2.1, 4.0, 5.2],
                "Ba2+": [0.8],
                "Be2+": [1.4, 2.5],
                "Bi3+": [0.0],
                "Ca2+": [1.4, 2.3],
                "Cd2+": [4.3, 7.8, 10.1],
                "Ce3+": [3.2, 5.8, 7.8],
                "Ce4+": [0.0],
                "Co2+": [4.3, 7.9, 10.8],
                "Cr3+": [5.1, 9.8, 13.9],
                "Cu2+": [8.2, 15.1],
                "Dy3+": [3.5, 6.4, 8.8],
                "Er3+": [3.6, 6.5, 9.0],
                "Eu3+": [3.4, 6.2, 8.5],
                "Fe2+": [4.3, 7.7, 10.0],
                "Fe3+": [0.0],
                "Ga3+": [4.2, 8.0, 11.1],
                "Gd3+": [3.4, 6.1, 8.4],
                "Hf4+": [0.0],
                "Hg2_2+": [0.0],
                "Hg2+": [10.3, 19.2],
                "Ho3+": [3.5, 6.4, 8.9],
                "In3+": [5.7, 10.7, 14.8],
                "La3+": [3.0, 5.4, 7.5],
                "Lu3+": [3.8, 6.9, 9.6],
                "Mg2+": [2.2, 3.9],
                "Mn2+": [2.8, 5.5, 8.2],
                "Nb5+": [0.0],
                "Nd3+": [3.2, 5.9, 8.2],
                "Ni2+": [5.4, 9.7, 14.2],
                "Pb2+": [4.8, 7.8],
                "Pr3+": [3.1, 5.6, 7.9],
                "Sc3+": [4.5, 8.4, 11.5],
                "Sm3+": [3.3, 6.0, 8.3],
                "Sn2+": [4.5, 8.1],
                "Sn4+": [0.0],
                "Sr2+": [1.0],
                "Ta5+": [0.0],
                "Tb3+": [3.4, 6.2, 8.6],
                "Th4+": [0.0],
                "Ti4+": [0.0],
                "Tl+": [0.9],
                "Tl3+": [6.5, 11.8, 15.5],
                "Tm3+": [3.7, 6.7, 9.2],
                "U4+": [0.0],
                "UO2_2+": [2.6, 4.4],
                "V3+": [0.0],
                "VO_2+": [3.2, 6.1],
                "Y3+": [3.5, 6.3, 8.6],
                "Yb3+": [3.8, 6.8, 9.4],
                "Zn2+": [5.0, 9.3, 12.1],
            }
        },

        # ---------------------------------------------------------------------
        # Citrate
        # ---------------------------------------------------------------------
        "Citrate": {
            "pkas": [3.13, 4.76, 6.40],

            "log_betas": {
                "Ag+": [0.9],
                "Al3+": [8.1, 14.2],
                "Ba2+": [2.5],
                "Be2+": [3.3],
                "Bi3+": [11.2],
                "Ca2+": [3.5],
                "Cd2+": [4.2, 6.5],
                "Ce3+": [7.5, 12.1],
                "Ce4+": [13.5],
                "Co2+": [5.0, 7.8],
                "Cr3+": [9.3],
                "Cu2+": [5.9, 10.2],
                "Dy3+": [7.9, 13.0],
                "Er3+": [8.1, 13.3],
                "Eu3+": [7.8, 12.8],
                "Fe2+": [4.4, 7.1],
                "Fe3+": [11.5, 18.5],
                "Ga3+": [10.0],
                "Gd3+": [7.8, 12.7],
                "Hf4+": [17.5],
                "Hg2_2+": [0.0],
                "Hg2+": [10.9],
                "Ho3+": [8.0, 13.2],
                "In3+": [10.2, 16.5],
                "La3+": [7.2, 11.5],
                "Lu3+": [8.5, 14.1],
                "Mg2+": [3.4],
                "Mn2+": [3.7, 5.8],
                "Nb5+": [0.0],
                "Nd3+": [7.6, 12.3],
                "Ni2+": [5.4, 8.4],
                "Pb2+": [5.5, 8.3],
                "Pr3+": [7.4, 11.9],
                "Sc3+": [10.5],
                "Sm3+": [7.7, 12.6],
                "Sn2+": [5.5],
                "Sn4+": [0.0],
                "Sr2+": [2.8],
                "Ta5+": [0.0],
                "Tb3+": [7.9, 12.9],
                "Th4+": [13.0, 22.2],
                "Ti4+": [14.2],
                "Tl+": [1.0],
                "Tl3+": [12.5],
                "Tm3+": [8.2, 13.5],
                "U4+": [14.5],
                "UO2_2+": [6.4, 11.1],
                "V3+": [7.8],
                "VO_2+": [6.8, 11.8],
                "Y3+": [7.8, 12.8],
                "Yb3+": [8.4, 13.9],
                "Zn2+": [5.0, 8.6],
            }
        },

        # ---------------------------------------------------------------------
        # Oxalate
        # ---------------------------------------------------------------------
        "Oxalate": {
            "pkas": [1.25, 4.14],

            "log_betas": {
                "Al3+": [6.1, 11.1, 15.1],
                "Be2+": [1.9, 3.2],
                "Cr3+": [5.3, 10.5, 15.2],
                "Fe3+": [7.5, 13.6, 18.5],
                "Hf4+": [10.2, 19.5, 27.5],
                "Nb5+": [9.5, 18.0, 25.0],
                "Ta5+": [9.0, 17.5, 24.0],
                "Ti4+": [8.5, 16.0],
                "V3+": [6.2, 11.5, 15.5],
                "VO_2+": [6.3, 11.4],
                "Zr4+": [10.0, 19.2, 27.2],
            }
        }
    }
}


# =============================================================================
# CHEMICAL DATABASE HELPERS
# =============================================================================

def get_available_metals():
    """Return all metals available in the hydroxide database."""
    return list(DATABASE["Hydroxide"]["Metals"].keys())


def get_ligand_pkas(ligand_name):
    """Return pKa values for the selected ligand."""
    return DATABASE["Chelators"][ligand_name]["pkas"]


def get_complex_parameters(metal_name, ligand_name):

    ligand_data = DATABASE["Chelators"][ligand_name]
    beta_list = ligand_data["log_betas"].get(metal_name)

    if not beta_list:
        return 0.0, 0, 0.0, False

    # A zero beta is explicitly treated as no effective complex formation.
    if beta_list[-1] <= 0.0:
        return 0.0, 0, beta_list[-1], False

    log_beta = beta_list[-1]
    coordination_number = len(beta_list)
    beta = 10.0 ** log_beta

    return beta, coordination_number, log_beta, True


# =============================================================================
# LIGAND PROTONATION
# =============================================================================

def calculate_inverse_alpha_Y(ph_value, pkas):

    h = 10.0 ** (-ph_value)

    ka_values = [10.0 ** (-pka) for pka in pkas]

    inverse_alpha = 1.0
    cumulative_product = 1.0

    for i, ka in enumerate(reversed(ka_values), start=1):
        cumulative_product *= ka
        inverse_alpha += (h ** i) / cumulative_product

    return inverse_alpha


# =============================================================================
# EQUILIBRIUM ENGINE
# =============================================================================

def solve_free_ligand(
    ph,
    metal_systems,
    ligand_name,
    total_ligand_concentration
):

    if total_ligand_concentration <= 0.0:
        return 0.0

    pkas = get_ligand_pkas(ligand_name)
    inverse_alpha = calculate_inverse_alpha_Y(ph, pkas)

    # Fully deprotonated ligand cannot exceed the total ligand concentration.
    low = 0.0
    high = total_ligand_concentration / inverse_alpha

    free_oh = 10.0 ** (ph - 14.0)

    for _ in range(150):

        free_ligand = 0.5 * (low + high)

        calculated_total = free_ligand * inverse_alpha

        for metal in metal_systems:

            metal_name = metal["name"]
            initial_conc = metal["initial_conc"]

            metal_data = DATABASE["Hydroxide"]["Metals"][metal_name]

            ksp = metal_data["ksp"]
            hydroxide_stoichiometry = metal_data["y"]

            beta, coordination_number, _, has_complex = (
                get_complex_parameters(metal_name, ligand_name)
            )

            if free_oh > 1.0e-30:
                ksp_solubility = ksp / (
                    free_oh ** hydroxide_stoichiometry
                )
            else:
                ksp_solubility = initial_conc

            if has_complex:
                complex_factor = (
                    1.0
                    + beta * (free_ligand ** coordination_number)
                )
            else:
                complex_factor = 1.0

            # Free metal concentration before precipitation.
            metal_free_unprecipitated = (
                initial_conc / complex_factor
            )

            # The free metal concentration cannot exceed the
            # hydroxide solubility ceiling.
            metal_free_real = min(
                metal_free_unprecipitated,
                ksp_solubility
            )

            if has_complex:
                complexed_metal = (
                    beta
                    * metal_free_real
                    * (free_ligand ** coordination_number)
                )

                calculated_total += (
                    coordination_number * complexed_metal
                )

        if calculated_total < total_ligand_concentration:
            low = free_ligand
        else:
            high = free_ligand

    return 0.5 * (low + high)


def calculate_precipitation_at_pH(
    ph,
    metal_systems,
    ligand_name=None,
    total_ligand_concentration=0.0
):
    """
    Calculate precipitation percentage for every selected metal at one pH.
    """

    ligand_active = (
        ligand_name is not None
        and total_ligand_concentration > 0.0
    )

    free_oh = 10.0 ** (ph - 14.0)

    if ligand_active:
        free_ligand = solve_free_ligand(
            ph,
            metal_systems,
            ligand_name,
            total_ligand_concentration
        )
    else:
        free_ligand = 0.0

    results = {
        "pH": ph,
        "free_ligand": free_ligand
    }

    for metal in metal_systems:

        metal_name = metal["name"]
        initial_conc = metal["initial_conc"]

        metal_data = DATABASE["Hydroxide"]["Metals"][metal_name]

        ksp = metal_data["ksp"]
        hydroxide_stoichiometry = metal_data["y"]

        if free_oh > 1.0e-30:
            ksp_solubility = ksp / (
                free_oh ** hydroxide_stoichiometry
            )
        else:
            ksp_solubility = initial_conc

        if ligand_active:
            beta, coordination_number, _, has_complex = (
                get_complex_parameters(
                    metal_name,
                    ligand_name
                )
            )
        else:
            beta = 0.0
            coordination_number = 0
            has_complex = False

        if has_complex:
            complex_factor = (
                1.0
                + beta * (free_ligand ** coordination_number)
            )
        else:
            complex_factor = 1.0

        metal_free_unprecipitated = (
            initial_conc / complex_factor
        )

        metal_free_real = min(
            metal_free_unprecipitated,
            ksp_solubility
        )

        # Total dissolved analytical metal:
        # free metal + complexed metal
        soluble_concentration = (
            metal_free_real * complex_factor
        )

        soluble_concentration = min(
            initial_conc,
            soluble_concentration
        )

        precipitation_percentage = (
            (initial_conc - soluble_concentration)
            / initial_conc
            * 100.0
        )

        results[metal_name] = float(
            np.clip(precipitation_percentage, 0.0, 100.0)
        )

    return results


# =============================================================================
# SIMULATION FUNCTIONS
# =============================================================================

def run_pH_simulation(
    metal_systems,
    ligand_name,
    ligand_concentration,
    ph_min,
    ph_max,
    number_of_points=1000
):
    """Run the full precipitation-vs-pH simulation."""

    ph_values = np.linspace(
        ph_min,
        ph_max,
        number_of_points
    )

    rows = []

    for ph in ph_values:

        result = calculate_precipitation_at_pH(
            ph=ph,
            metal_systems=metal_systems,
            ligand_name=ligand_name,
            total_ligand_concentration=ligand_concentration
        )

        rows.append(result)

    df = pd.DataFrame(rows)

    return df


def run_ligand_simulation(
    metal_systems,
    ligand_name,
    fixed_pH,
    ligand_min,
    ligand_max,
    number_of_points=300
):
    """
    Run precipitation as a function of total ligand concentration
    at a fixed pH.

    The ligand concentration is sampled logarithmically because
    complexation effects can span many orders of magnitude.
    """

    if ligand_min <= 0.0:
        ligand_min = 1.0e-12

    if ligand_max <= ligand_min:
        ligand_max = ligand_min * 1.0e6

    ligand_values = np.logspace(
        np.log10(ligand_min),
        np.log10(ligand_max),
        number_of_points
    )

    rows = []

    for ligand_concentration in ligand_values:

        result = calculate_precipitation_at_pH(
            ph=fixed_pH,
            metal_systems=metal_systems,
            ligand_name=ligand_name,
            total_ligand_concentration=ligand_concentration
        )

        result["ligand_concentration"] = ligand_concentration

        rows.append(result)

    return pd.DataFrame(rows)


# =============================================================================
# VISUALIZATION
# =============================================================================

def smooth_series(values):
    """
    Apply a mild Savitzky-Golay filter for visualization.

    This does NOT modify the underlying simulation data.
    """

    values = np.asarray(values)

    if len(values) < 11:
        return values

    try:
        smoothed = savgol_filter(
            values,
            window_length=11,
            polyorder=2
        )

        return np.clip(smoothed, 0.0, 100.0)

    except Exception:
        return values


def create_pH_plot(
    df,
    selected_metals,
    ph_min,
    ph_max,
    ligand_name,
    ligand_active
):
    """Create precipitation percentage vs pH plot."""

    fig, ax = plt.subplots(figsize=(10, 6))

    for metal in selected_metals:

        y_values = smooth_series(
            df[metal]
        )

        ax.plot(
            df["pH"],
            y_values,
            label=metal,
            linewidth=2.5
        )

    if ligand_active:
        title = (
            f"Competitive Hydroxide Precipitation\n"
            f"with {ligand_name}"
        )
    else:
        title = "Hydroxide Precipitation without Chelating Agent"

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(
        "pH",
        fontsize=12
    )

    ax.set_ylabel(
        "Precipitated Metal (%)",
        fontsize=12
    )

    ax.set_xlim(
        ph_min,
        ph_max
    )

    ax.set_ylim(
        -2,
        102
    )

    ax.set_xticks(
        np.arange(
            np.ceil(ph_min),
            np.floor(ph_max) + 1,
            1
        )
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.5
    )

    ax.legend(
        loc="best"
    )

    fig.tight_layout()

    return fig


def create_ligand_plot(
    df,
    selected_metals,
    fixed_pH,
    ligand_name
):
    """Create precipitation percentage vs ligand concentration plot."""

    fig, ax = plt.subplots(figsize=(10, 6))

    for metal in selected_metals:

        y_values = smooth_series(
            df[metal]
        )

        ax.plot(
            df["ligand_concentration"],
            y_values,
            label=metal,
            linewidth=2.5
        )


    ax.set_title(
        f"Effect of {ligand_name} Concentration\n"
        f"at Fixed pH = {fixed_pH:.2f}",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(
        f"Total {ligand_name} Concentration (M)",
        fontsize=12
    )

    ax.set_ylabel(
        "Precipitated Metal (%)",
        fontsize=12
    )

    ax.set_ylim(
        -2,
        102
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.5,
        which="both"
    )

    ax.legend(
        loc="best"
    )

    fig.tight_layout()

    return fig


# =============================================================================
# STREAMLIT INTERFACE
# =============================================================================

st.set_page_config(
    page_title="Hydroxide precipitation simulator with chelating agent",
    page_icon="🧪",
    layout="wide"
)


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------

st.title(
    "🧪 Hydroxide precipitation simulator with chelating agent"
)

st.markdown(
    """
This tool estimates metal hydroxide precipitation under idealized
thermodynamic conditions, with optional competition from a chelating agent.

The chemical constants are retrieved automatically from the built-in
database. You do **not** need to enter Ksp, pKa, or formation constants manually.
"""
)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

st.sidebar.header("Simulation Settings")


# Chelating agent
ligand_options = list(
    DATABASE["Chelators"].keys()
)

ligand_name = st.sidebar.selectbox(
    "Chelating agent",
    options=ligand_options
)

use_ligand = st.sidebar.checkbox(
    "Include chelating agent",
    value=True
)


# -----------------------------------------------------------------------------
# Metal selection
# -----------------------------------------------------------------------------

st.sidebar.subheader("Metal Selection")

available_metals = get_available_metals()

selected_metals = st.sidebar.multiselect(
    "Select metals",
    options=available_metals,
    default=["Al3+", "Fe3+"]
)


if not selected_metals:

    st.warning(
        "Please select at least one metal."
    )

    st.stop()


# -----------------------------------------------------------------------------
# Metal concentrations
# -----------------------------------------------------------------------------

st.sidebar.subheader(
    "Initial Metal Concentrations"
)

metal_systems = []

for metal_name in selected_metals:

    concentration = st.sidebar.number_input(
        f"{metal_name} concentration (M)",
        min_value=1.0e-12,
        max_value=1000.0,
        value=0.05 if metal_name == "Fe3+" else 0.50,
        format="%.6g",
        key=f"conc_{metal_name}"
    )

    metal_systems.append({
        "name": metal_name,
        "initial_conc": concentration
    })


# -----------------------------------------------------------------------------
# Ligand concentration
# -----------------------------------------------------------------------------

if use_ligand:

    st.sidebar.subheader(
        "Chelating Agent"
    )

    ligand_concentration = st.sidebar.number_input(
        f"Total {ligand_name} concentration (M)",
        min_value=0.0,
        max_value=1000.0,
        value=0.20,
        format="%.6g"
    )

else:

    ligand_concentration = 0.0


# -----------------------------------------------------------------------------
# pH range
# -----------------------------------------------------------------------------

st.sidebar.subheader(
    "pH Simulation Range"
)

ph_min = st.sidebar.slider(
    "Minimum pH",
    min_value=0.0,
    max_value=14.0,
    value=0.0,
    step=0.1
)

ph_max = st.sidebar.slider(
    "Maximum pH",
    min_value=0.0,
    max_value=14.0,
    value=14.0,
    step=0.1
)

if ph_max <= ph_min:

    st.error(
        "Maximum pH must be greater than minimum pH."
    )

    st.stop()


# -----------------------------------------------------------------------------
# Second graph settings
# -----------------------------------------------------------------------------

st.sidebar.subheader(
    "Fixed-pH Ligand Analysis"
)

fixed_pH = st.sidebar.slider(
    "Fixed pH",
    min_value=0.0,
    max_value=14.0,
    value=7.0,
    step=0.1
)

ligand_min = st.sidebar.number_input(
    "Minimum ligand concentration (M)",
    min_value=1.0e-12,
    max_value=1000.0,
    value=1.0e-6,
    format="%.3e"
)

ligand_max = st.sidebar.number_input(
    "Maximum ligand concentration (M)",
    min_value=1.0e-11,
    max_value=1000.0,
    value=1.0,
    format="%.3e"
)


# =============================================================================
# DATABASE INFORMATION
# =============================================================================

st.subheader("Selected Chemical System")

info_columns = st.columns(
    min(len(selected_metals), 4)
)

for i, metal_name in enumerate(selected_metals):

    metal_data = DATABASE["Hydroxide"]["Metals"][metal_name]

    with info_columns[i % len(info_columns)]:

        st.markdown(
            f"### {metal_name}"
        )

        st.caption(
            f"Ksp = {metal_data['ksp']:.3e}"
        )

        st.caption(
            f"Hydroxide stoichiometry: M(OH)₍{metal_data['y']}₎"
        )

        if use_ligand:

            (
                _,
                coordination_number,
                log_beta,
                has_complex
            ) = get_complex_parameters(
                metal_name,
                ligand_name
            )

            if has_complex:

                st.caption(
                    f"{ligand_name} complex: "
                    f"M(L)₍{coordination_number}₎"
                )

                st.caption(
                    f"log β = {log_beta:.2f}"
                )

            else:

                st.caption(
                    f"No stable modeled {metal_name}-{ligand_name} "
                    "complex in database"
                )


# =============================================================================
# RUN SIMULATION
# =============================================================================

st.divider()

run_simulation = st.button(
    "▶ Run Simulation",
    type="primary",
    use_container_width=True
)


if run_simulation:

    # -------------------------------------------------------------------------
    # pH simulation
    # -------------------------------------------------------------------------

    with st.spinner(
        "Running pH equilibrium simulation..."
    ):

        df_ph = run_pH_simulation(
            metal_systems=metal_systems,
            ligand_name=ligand_name if use_ligand else None,
            ligand_concentration=(
                ligand_concentration if use_ligand else 0.0
            ),
            ph_min=ph_min,
            ph_max=ph_max
        )


    # -------------------------------------------------------------------------
    # Ligand concentration simulation
    # -------------------------------------------------------------------------

    with st.spinner(
        "Running ligand concentration simulation..."
    ):

        df_ligand = run_ligand_simulation(
            metal_systems=metal_systems,
            ligand_name=ligand_name,
            fixed_pH=fixed_pH,
            ligand_min=ligand_min,
            ligand_max=ligand_max
        )


    # =========================================================================
    # GRAPH 1 — PRECIPITATION VS pH
    # =========================================================================

    st.subheader(
        "1. Hydroxide Precipitation vs pH"
    )

    fig_ph = create_pH_plot(
        df=df_ph,
        selected_metals=selected_metals,
        ph_min=ph_min,
        ph_max=ph_max,
        ligand_name=ligand_name,
        ligand_active=use_ligand
    )

    st.pyplot(
        fig_ph,
        use_container_width=True
    )

    plt.close(fig_ph)


    # =========================================================================
    # GRAPH 2 — PRECIPITATION VS LIGAND CONCENTRATION
    # =========================================================================

    st.subheader(
        "2. Effect of Chelating Agent Concentration"
    )

    if use_ligand:

        fig_ligand = create_ligand_plot(
            df=df_ligand,
            selected_metals=selected_metals,
            fixed_pH=fixed_pH,
            ligand_name=ligand_name
        )

        st.pyplot(
            fig_ligand,
            use_container_width=True
        )

        plt.close(fig_ligand)

    else:

        st.info(
            "Enable the chelating agent to analyze the effect "
            "of ligand concentration."
        )


    # =========================================================================
    # RESULTS TABLE
    # =========================================================================

    with st.expander(
        "Show numerical results"
    ):

        st.dataframe(
            df_ph,
            use_container_width=True
        )
