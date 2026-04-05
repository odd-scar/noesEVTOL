import sys
import os
sys.stdout.reconfigure(encoding="utf-8")   # needed for box-drawing chars on Windows

import numpy as np
import matplotlib
try:
    matplotlib.use("TkAgg")          # preferred for interactive use
except Exception:
    matplotlib.use("Agg")            # fallback for headless / no-tk environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import Optional
from pint import UnitRegistry

# ============================================================
# Unit setup
# ============================================================
ureg = UnitRegistry()
Q_ = ureg.Quantity

ft      = ureg.ft
s       = ureg.s
rad     = ureg.radian
rpm     = ureg.rpm
slug    = ureg.slug
kts     = ureg.knot
nmi     = ureg.nautical_mile
hp      = ureg.horsepower
lbf     = ureg.lbf
kg      = ureg.kilogram
kWh     = ureg.kilowatt_hour
minute  = ureg.minute
hour    = ureg.hour

g_si = 9.81  * ureg.m  / s**2   # SI gravity
g_ft = 32.174 * ft / s**2        # imperial gravity

# ============================================================
# ISA Standard Atmosphere
# ============================================================
RHO_SL = 0.002377   # slug/ft³  (reference)


def isa_density(h_ft: float) -> float:
    """
    Air density in slug/ft³ via ISA model.
    Troposphere  0–36 089 ft : T lapse at 3.566 2e-3 °R/ft
    Stratosphere >36 089 ft  : isothermal, exponential decay
    """
    T0   = 518.67        # °R   sea-level temperature
    rho0 = 0.002377      # slug/ft³
    L    = 3.5662e-3     # °R/ft  lapse rate
    h_tp = 36_089.0      # ft  tropopause

    if h_ft <= h_tp:
        T = T0 - L * h_ft
        return rho0 * (T / T0) ** 4.2561
    T_tp   = T0 - L * h_tp
    rho_tp = rho0 * (T_tp / T0) ** 4.2561
    return rho_tp * np.exp(-4.80634e-5 * (h_ft - h_tp))


def isa_speed_of_sound(h_ft: float) -> float:
    """Speed of sound in ft/s at altitude h_ft."""
    T0   = 518.67
    L    = 3.5662e-3
    h_tp = 36_089.0
    T    = max(T0 - L * h_ft, T0 - L * h_tp)
    return np.sqrt(1.4 * 1716.0 * T)   # gamma * R_air * T


# ============================================================
# CSV Standard Atmosphere  (standard_atmosphere.csv in same folder)
# Used for performance sweeps (ROC, max speed) — more accurate than
# the two-layer ISA formula for intermediate altitudes.
#
# CSV columns (all metric):
#   0: height [m]          → converted to ft  (× 3.28084)
#   1: density [kg/m³]     → converted to slug/ft³  (÷ 515.379)
#   2: pressure [N/m²]     (not used here)
#   3: temperature [K]     (stored for reference)
#   4: speed_of_sound [m/s]→ converted to ft/s (× 3.28084)
#   5: dynamic_viscosity    (not used here)
#   6: kinematic_viscosity  (not used here)
# ============================================================
def _load_csv_atmosphere() -> dict:
    """
    Load standard_atmosphere.csv and return a dict of arrays
    all sorted by altitude in ft.
    """
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "standard_atmosphere.csv")
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    h_m        = data[:, 0]
    rho_kg_m3  = data[:, 1]
    temp_K     = data[:, 3]
    sos_m_s    = data[:, 4]

    alt_ft      = h_m       * 3.28084          # m → ft
    rho_sl_ft3  = rho_kg_m3 / 515.379          # kg/m³ → slug/ft³
    sos_ft_s    = sos_m_s   * 3.28084          # m/s → ft/s

    idx = np.argsort(alt_ft)
    return {
        "alt_ft":       alt_ft[idx],
        "rho_slug_ft3": rho_sl_ft3[idx],
        "temp_K":       temp_K[idx],
        "sos_ft_s":     sos_ft_s[idx],
    }


_CSV_ATM = _load_csv_atmosphere()

# Keep backwards-compatible names used elsewhere
_CSV_ALT_FT       = _CSV_ATM["alt_ft"]
_CSV_RHO_SLUG_FT3 = _CSV_ATM["rho_slug_ft3"]


def csv_density(h_ft: float) -> float:
    """
    Air density in slug/ft³ interpolated from standard_atmosphere.csv.
    Used in ROC and max-speed performance sweeps.
    """
    return float(np.interp(h_ft, _CSV_ATM["alt_ft"], _CSV_ATM["rho_slug_ft3"]))


def csv_speed_of_sound(h_ft: float) -> float:
    """
    Speed of sound in ft/s interpolated from standard_atmosphere.csv.
    More consistent than the 2-layer ISA formula when sweeping with CSV density.
    """
    return float(np.interp(h_ft, _CSV_ATM["alt_ft"], _CSV_ATM["sos_ft_s"]))


def csv_sigma(h_ft: float) -> float:
    """
    Density ratio σ = ρ(h) / ρ_SL  from CSV atmosphere.
    Used for power-available lapse at altitude.
    """
    rho_sl = float(np.interp(0.0, _CSV_ATM["alt_ft"], _CSV_ATM["rho_slug_ft3"]))
    rho_h  = float(np.interp(h_ft, _CSV_ATM["alt_ft"], _CSV_ATM["rho_slug_ft3"]))
    return rho_h / rho_sl


# ============================================================
# Dataclasses
# ============================================================
@dataclass
class RotorGeometry:
    radius:   Q_
    chord:    Q_
    n_blades: int
    n_wheels: int


@dataclass
class FuselageGeometry:
    cockpit_length: Q_
    cockpit_width:  Q_
    cockpit_height: Q_
    cabin_length:   Q_
    cabin_width:    Q_
    cabin_height:   Q_
    landing_gear_height: Q_   # height of gear strut above ground

    @property
    def fuselage_length(self) -> Q_:
        return self.cockpit_length + self.cabin_length

    @property
    def wetted_area_box_estimate(self) -> Q_:
        """Rough box-model wetted area (ft²)."""
        cockpit = 2 * (
            self.cockpit_length * self.cockpit_width
            + self.cockpit_length * self.cockpit_height
            + self.cockpit_width  * self.cockpit_height
        )
        cabin = 2 * (
            self.cabin_length * self.cabin_width
            + self.cabin_length * self.cabin_height
            + self.cabin_width  * self.cabin_height
        )
        return (cockpit + cabin).to(ft**2)


@dataclass
class PayloadWeights:
    crew_and_passengers: Q_
    electronics:         Q_
    defense_system:      Q_

    @property
    def total_payload(self) -> Q_:
        return (self.crew_and_passengers
                + self.electronics
                + self.defense_system).to(lbf)


@dataclass
class AeroCoefficients:
    cd0_hover:            float
    induced_power_factor: float = 1.15
    profile_power_factor: float = 4.7
    power_margin:         float = 1.25   # P_installed / P_hover_SL


@dataclass
class BatteryModel:
    specific_energy:  Q_          # kWh/kg
    usable_fraction:  float = 0.90


@dataclass
class MissionProfile:
    cruise_range:        Q_
    cruise_altitude:     Q_
    climb_rate:          Q_
    cruise_speed:        Q_
    descent_rate:        Q_
    hover_time_takeoff:  Q_   # hover before climbing
    hover_time_landing:  Q_   # hover after descending


@dataclass
class RotorcraftDesign:
    rotor:                  RotorGeometry
    fuselage:               FuselageGeometry
    payload:                PayloadWeights
    aero:                   AeroCoefficients
    battery:                BatteryModel
    gw_guess:               Q_
    hover_rotor_speed_rpm:  float = 300.0   # RPM used for hover / climb / descent


# ============================================================
# Utility functions
# ============================================================
def to_omega(rotor_speed: Q_) -> Q_:
    """
    Convert rotor speed to angular velocity in 1/s.
    We deliberately avoid 'rad/s' because pint 0.17+ tracks radians as a
    unit with dimension, which causes CT to carry rad^2 and overflow.
    Radians are dimensionless, so 1/s is numerically identical.
    """
    omega_val = rotor_speed.to(rpm).magnitude * (2.0 * np.pi / 60.0)
    return omega_val / s


def omega_from_tip_speed(tip_speed: Q_, radius: Q_) -> Q_:
    """Convert tip speed + radius to angular velocity in 1/s (rad-free)."""
    return (tip_speed / radius).to(1 / s)


def tip_speed_schedule(V_inf: Q_) -> Q_:
    """
    Empirical rotor tip-speed schedule vs forward speed.
    Adjust V_pts / tip_pts to match your vehicle's data.
    """
    V_kts  = V_inf.to(kts).magnitude
    V_pts  = np.array([0,   40,  80,  120, 130, 150, 170])   # kts
    tip_pts = np.array([400, 500, 500, 400, 500, 550, 600])   # ft/s
    return np.interp(V_kts, V_pts, tip_pts) * ft / s


def rotor_disk_area(radius: Q_) -> Q_:
    return (np.pi * radius**2).to(ft**2)


def solidity(n_blades: int, chord: Q_, radius: Q_) -> float:
    c_ = float(chord.to(ft).magnitude)
    R_ = float(radius.to(ft).magnitude)
    return n_blades * c_ / (np.pi * R_)


def thrust_coefficient(T: Q_, rho: Q_, radius: Q_, omega: Q_) -> float:
    """CT computed entirely in slug/ft/s — avoids pint unit-chain issues."""
    T_   = float(T.to(lbf).magnitude)
    rho_ = float(rho.to(slug / ft**3).magnitude)
    R_   = float(radius.to(ft).magnitude)
    om_  = float(omega.to(1 / s).magnitude)
    A_   = np.pi * R_**2
    return T_ / (rho_ * A_ * (om_ * R_)**2)


def advance_ratio(V_inf: Q_, alpha: Q_, omega: Q_, radius: Q_) -> float:
    """μ computed from magnitudes in ft/s — avoids pint unit-chain issues."""
    V_  = float(V_inf.to(ft / s).magnitude)
    om_ = float(omega.to(1 / s).magnitude)
    R_  = float(radius.to(ft).magnitude)
    a_  = float(alpha.to(rad).magnitude)
    return V_ * np.cos(a_) / (om_ * R_)


def thrust_from_weight_and_tilt(weight: Q_, alpha: Q_) -> Q_:
    """
    Cruise thrust: rotor tilted forward by alpha.
    Vertical equilibrium → T·cos(α) = W → T = W / cos(α).
    (Previous version used W*cos(α) which was incorrect.)
    """
    return (weight / np.cos(alpha.to(rad).magnitude)).to(lbf)


# ============================================================
# Hover / axial power
# ============================================================
def hover_induced_velocity(T: Q_, rho: Q_, radius: Q_) -> Q_:
    """Induced velocity in ft/s using magnitude arithmetic."""
    T_   = float(T.to(lbf).magnitude)
    rho_ = float(rho.to(slug / ft**3).magnitude)
    R_   = float(radius.to(ft).magnitude)
    A_   = np.pi * R_**2
    vi   = float(np.sqrt(T_ / (2.0 * rho_ * A_)))   # ft/s
    return vi * ft / s


def hover_power_coefficients(design: RotorcraftDesign,
                              T: Q_, rho: Q_, omega: Q_) -> dict:
    sigma      = solidity(design.rotor.n_blades, design.rotor.chord, design.rotor.radius)
    ct         = thrust_coefficient(T, rho, design.rotor.radius, omega)
    cp_induced = design.aero.induced_power_factor * ct**(3/2) / np.sqrt(2)
    cp_profile = design.aero.cd0_hover * sigma / 8
    return {
        "sigma":       sigma,
        "C_T":         ct,
        "C_P_induced": cp_induced,
        "C_P_profile": cp_profile,
        "C_P_hover":   cp_induced + cp_profile,
    }


def hover_power_required(cp_hover: float, rho: Q_,
                          radius: Q_, omega: Q_) -> Q_:
    """Power in hp using magnitude arithmetic (avoids pint unit-chain issues)."""
    rho_ = float(rho.to(slug / ft**3).magnitude)
    R_   = float(radius.to(ft).magnitude)
    om_  = float(omega.to(1 / s).magnitude)
    A_   = np.pi * R_**2
    Vtip = om_ * R_
    P_ftlbfs = cp_hover * rho_ * A_ * Vtip**3   # ft·lbf/s
    return float(P_ftlbfs / 550.0) * hp          # 1 hp = 550 ft·lbf/s


def axial_power_required(p_hover: Q_, vertical_speed: Q_,
                          vi_hover: Q_,
                          k_descent: float = 1.15) -> dict:
    """
    Power for pure-axial climb or descent via momentum theory.
    vertical_speed positive = climb, negative = descent.
    """
    ratio = float(vertical_speed.to(ft / s).magnitude) / float(vi_hover.to(ft / s).magnitude)

    # Climb
    mult_climb = ratio / 2 + np.sqrt((ratio / 2)**2 + 1)
    p_climb    = (mult_climb * p_hover).to(hp)

    # Slow descent (momentum-theory polynomial fit)
    mult_slow_desc = (ratio
                      + (k_descent
                         - 1.125 * ratio
                         - 1.372 * ratio**2
                         - 1.718 * ratio**3
                         - 0.655 * ratio**4))
    p_slow_desc = (mult_slow_desc * p_hover).to(hp)

    # Fast descent (windmill state, |ratio| >= 2)
    fast_valid = ratio <= -2.0
    if fast_valid:
        mult_fast = ratio / 2 - np.sqrt((ratio / 2)**2 - 1)
        p_fast    = (mult_fast * p_hover).to(hp)
    else:
        p_fast = np.nan * hp

    return {
        "ratio_Vc_over_Vh":    ratio,
        "P_climb_axial":       p_climb,
        "P_slow_descent_axial": p_slow_desc,
        "P_fast_descent_axial": p_fast,
        "fast_descent_valid":  fast_valid,
    }


# ============================================================
# Forward-flight power
# ============================================================
def fuselage_parasite_drag_area(weight: Q_, radius: Q_,
                                 category: str = "clean") -> dict:
    W_lbf = weight.to(lbf).magnitude
    if category == "clean":
        f = 0.4 * np.sqrt(W_lbf) * ft**2
    elif category == "utility":
        f = 0.8 * np.sqrt(W_lbf) * ft**2
    else:
        raise ValueError("category must be 'clean' or 'utility'")
    A = rotor_disk_area(radius)
    return {
        "flat_plate_area":       f.to(ft**2),
        "flat_plate_area_dimless": float(f.to(ft**2).magnitude) / float(A.to(ft**2).magnitude),
    }


def forward_flight_power_coefficient(
        design:           RotorcraftDesign,
        weight:           Q_,
        T:                Q_,
        rho:              Q_,
        omega:            Q_,
        V_inf:            Q_,
        alpha:            Q_,
        climb_rate:       Q_,
        parasite_category: str = "clean") -> dict:

    # Extract all magnitudes in consistent imperial units
    rho_ = float(rho.to(slug / ft**3).magnitude)
    R_   = float(design.rotor.radius.to(ft).magnitude)
    om_  = float(omega.to(1 / s).magnitude)
    Vc_  = float(climb_rate.to(ft / s).magnitude)
    W_   = float(weight.to(lbf).magnitude)
    A_   = np.pi * R_**2
    Vtip = om_ * R_

    sigma   = solidity(design.rotor.n_blades, design.rotor.chord, design.rotor.radius)
    ct      = thrust_coefficient(T, rho, design.rotor.radius, omega)
    mu      = advance_ratio(V_inf, alpha, omega, design.rotor.radius)

    if mu <= 0:
        raise ValueError("Forward-flight model requires mu > 0.")

    par     = fuselage_parasite_drag_area(weight, design.rotor.radius, parasite_category)

    cp_i    = design.aero.induced_power_factor * ct**2 / (2 * mu)
    cp_pr   = (design.aero.cd0_hover * sigma / 8
               * (1 + design.aero.profile_power_factor * mu**2))
    cp_par  = 0.5 * par["flat_plate_area_dimless"] * mu**3
    cp_cl   = float((Vc_ * W_) / (rho_ * A_ * Vtip**3))  # all in slug/ft/s system
    cp_tot  = cp_i + cp_pr + cp_par + cp_cl

    return {
        "mu":                    mu,
        "C_T":                   ct,
        "C_P_induced_fwd":       cp_i,
        "C_P_profile_fwd":       cp_pr,
        "C_P_parasitic":         cp_par,
        "C_P_climb":             cp_cl,
        "C_P_forward_total":     cp_tot,
        "flat_plate_area":       par["flat_plate_area"],
        "flat_plate_area_dimless": par["flat_plate_area_dimless"],
    }


def forward_flight_power_required(cp_forward: float, rho: Q_,
                                   radius: Q_, omega: Q_) -> Q_:
    rho_ = float(rho.to(slug / ft**3).magnitude)
    R_   = float(radius.to(ft).magnitude)
    om_  = float(omega.to(1 / s).magnitude)
    A_   = np.pi * R_**2
    Vtip = om_ * R_
    P_ftlbfs = cp_forward * rho_ * A_ * Vtip**3
    return float(P_ftlbfs / 550.0) * hp


# ============================================================
# Mission energy & battery
# ============================================================
def mission_energy(mission: MissionProfile,
                   battery:  BatteryModel,
                   p_hover:   Q_,
                   p_climb:   Q_,
                   p_cruise:  Q_,
                   p_descent: Q_) -> dict:
    """
    Energy breakdown for:
      hover-takeoff → climb → cruise → descent → hover-landing
    """
    t_hov_to   = mission.hover_time_takeoff.to(s)
    t_climb    = (mission.cruise_altitude / mission.climb_rate).to(s)
    t_cruise   = (mission.cruise_range    / mission.cruise_speed).to(s)
    t_descent  = (mission.cruise_altitude / mission.descent_rate).to(s)
    t_hov_land = mission.hover_time_landing.to(s)

    E_hov_to   = (p_hover   * t_hov_to).to(kWh)
    E_climb    = (p_climb   * t_climb).to(kWh)
    E_cruise   = (p_cruise  * t_cruise).to(kWh)
    E_descent  = (p_descent * t_descent).to(kWh)
    E_hov_land = (p_hover   * t_hov_land).to(kWh)

    E_req  = (E_hov_to + E_climb + E_cruise + E_descent + E_hov_land).to(kWh)
    E_nom  = (E_req / battery.usable_fraction).to(kWh)

    return {
        "t_hover_takeoff":  t_hov_to,
        "t_climb":          t_climb,
        "t_cruise":         t_cruise,
        "t_descent":        t_descent,
        "t_hover_landing":  t_hov_land,
        "E_hover_takeoff":  E_hov_to,
        "E_climb":          E_climb,
        "E_cruise":         E_cruise,
        "E_descent":        E_descent,
        "E_hover_landing":  E_hov_land,
        "E_required":       E_req,
        "E_nominal":        E_nom,
    }


def battery_from_energy(E_nominal: Q_, battery: BatteryModel) -> dict:
    m_batt_slug = (E_nominal / battery.specific_energy).to(slug)
    w_batt      = (m_batt_slug * g_ft).to(lbf)
    return {"weight": w_batt}


# ============================================================
# Empirical component weight equations
# ============================================================
def component_weight_equations(rotor:              RotorGeometry,
                                gw_guess:         Q_,
                                fuselage_length:    Q_,
                                fuselage_wetted_area: Q_,
                                omega:              Q_) -> dict:
    """
    Helicopter empirical equations + fraction-based groups for
    electric-specific components not covered by classic equations.

    Explicit equations  (Prouty-style, inputs in ft / lbf):
      main rotor blades, fuselage body, landing gear, cockpit controls,
      instruments, furnishings, air conditioning.

    Fraction-based groups (% of MTOW) to close the empty-weight budget:
      propulsion (motors+ESCs), drive system, electrical, flight controls,
      avionics, structural margin (booms, frames, misc not in Prouty).
    """
    b    = rotor.n_blades
    c    = rotor.chord.to(ft).magnitude
    R    = rotor.radius.to(ft).magnitude
    W    = gw_guess.to(lbf).magnitude
    Lf   = fuselage_length.to(ft).magnitude
    Swet = fuselage_wetted_area.to(ft**2).magnitude
    Nw   = rotor.n_wheels
    Vtip = (omega * rotor.radius).to(ft / s).magnitude

    return {
        # --- Prouty-style explicit equations ---
        "w_main_rotor_blades":  (0.026 * b**0.66 * c * R**1.3 * Vtip**0.67)   * lbf,
        "w_body_fuselage":      (6.9   * (W/1000)**0.49 * Lf**0.61 * Swet**0.25) * lbf,
        "w_landing_gear":       (40    * (W/1000)**0.67 * Nw**0.54)            * lbf,
        "w_cockpit_controls":   (11.5  * (W/1000)**0.40)                       * lbf,
        "w_instruments":        (3.5   * (W/1000)**1.30)                       * lbf,
        "w_furnishings":        (13.0  * (W/1000)**1.30)                       * lbf,
        "w_air_conditioning":   (8.0   * (W/1000))                             * lbf,
        # --- Fraction-based groups (eVTOL additions) ---
        "w_propulsion_motors":  (0.060 * W) * lbf,   # motors + ESCs
        "w_drive_system":       (0.030 * W) * lbf,   # minimal for direct-drive electric
        "w_electrical":         (0.040 * W) * lbf,   # wiring, bus, BMS
        "w_flight_controls":    (0.030 * W) * lbf,   # actuators, FBW computers
        "w_avionics":           (0.010 * W) * lbf,   # base avionics (excl. 1000 lb payload bay)
        "w_structural_margin":  (0.150 * W) * lbf,   # booms, frames, misc not in Prouty
    }


def sum_component_weights(cw: dict) -> Q_:
    total = 0.0 * lbf
    for v in cw.values():
        total += v.to(lbf)
    return total.to(lbf)


# ============================================================
# Fuselage volume / sizing check
# Reference formulas:
#   Cockpit_Volume = Lc * Wc * Hc
#   Cabin_Volume   = Lcab * Wcab * Hcab
#   V_usable       = Cockpit_Volume + Cabin_Volume
#   V_seats_tot    = N_occupants * (L_seat * W_seat * H_seat)
#   V_electronics  = K_elec * W_electronics  [K = ft³/lb]
#   V_payload      = V_seats_tot + V_electronics
#   V_energy_src   = K_batt * W_battery      [K = ft³/lb]
#   V_margin       = V_usable - (V_energy_src + V_payload)
#   V_margin >= 0 → fits;  < 0 → does not fit
# ============================================================
def fuselage_volume_check(design: "RotorcraftDesign",
                           w_battery: Q_,
                           n_occupants:          int   = 10,
                           seat_l_ft:            float = 2.5,
                           seat_w_ft:            float = 1.5,
                           seat_h_ft:            float = 3.0,
                           k_electronics_ft3_lb: float = 0.020,
                           k_battery_ft3_lb:     float = 0.025) -> dict:
    """
    Volume sizing check — verifies the fuselage can physically contain
    the payload (seats + electronics bay) and the battery pack.

    Parameters
    ----------
    design               : RotorcraftDesign dataclass
    w_battery            : computed battery weight (lbf Quantity)
    n_occupants          : total crew + passengers (default 10)
    seat_l/w/h_ft        : per-seat envelope dimensions (ft)
    k_electronics_ft3_lb : electronics volume density (ft³/lb)
    k_battery_ft3_lb     : battery volume density    (ft³/lb)

    Returns dict with all intermediate volumes and V_margin.
    """
    fus = design.fuselage
    Lc  = fus.cockpit_length.to(ft).magnitude
    Wc  = fus.cockpit_width.to(ft).magnitude
    Hc  = fus.cockpit_height.to(ft).magnitude
    Lca = fus.cabin_length.to(ft).magnitude
    Wca = fus.cabin_width.to(ft).magnitude
    Hca = fus.cabin_height.to(ft).magnitude

    # Usable interior volume
    V_cockpit = Lc * Wc * Hc
    V_cabin   = Lca * Wca * Hca
    V_usable  = V_cockpit + V_cabin

    # Payload volumes
    V_seats   = n_occupants * (seat_l_ft * seat_w_ft * seat_h_ft)
    W_elec    = design.payload.electronics.to(lbf).magnitude
    V_elec    = k_electronics_ft3_lb * W_elec
    V_payload = V_seats + V_elec

    # Energy source volume
    W_batt    = w_battery.to(lbf).magnitude
    V_batt    = k_battery_ft3_lb * W_batt

    V_margin  = V_usable - (V_batt + V_payload)

    return {
        "V_cockpit_ft3":  V_cockpit,
        "V_cabin_ft3":    V_cabin,
        "V_usable_ft3":   V_usable,
        "V_seats_ft3":    V_seats,
        "V_electronics_ft3": V_elec,
        "V_payload_ft3":  V_payload,
        "V_battery_ft3":  V_batt,
        "V_margin_ft3":   V_margin,
        "fits":           V_margin >= 0.0,
    }


# ============================================================
# CG calculation (simplified geometric shapes)
# ============================================================
def component_cg_locations(design: RotorcraftDesign) -> dict:
    """
    Returns {component_name: (x_ft, z_ft)} for every weight group.

    Coordinate system
    -----------------
    x = 0 at nose, positive aft
    z = 0 at ground, positive up

    Shapes assumed
    -------
    fuselage  → rectangular box  (CG at box centroid)
    rotor hub → circular disk    (CG at hub centre, above fuselage)
    landing gear → vertical strut (CG at strut mid-point)
    batteries → floor slab       (CG at floor level, aft-centre of cabin)
    passengers → seated human    (CG at seat cushion height, cabin mid)
    """
    f   = design.fuselage
    gl  = f.landing_gear_height.to(ft).magnitude   # gear strut height
    ckl = f.cockpit_length.to(ft).magnitude
    cbl = f.cabin_length.to(ft).magnitude
    fsl = f.fuselage_length.to(ft).magnitude
    ckh = f.cockpit_height.to(ft).magnitude
    cbh = f.cabin_height.to(ft).magnitude
    fsh = max(ckh, cbh)                            # effective fuselage height

    # All positions: (x along fuselage from nose, z above ground)
    return {
        "w_main_rotor_blades":  (fsl * 0.45,        gl + fsh + 1.0),
        "w_body_fuselage":      (fsl * 0.50,         gl + fsh * 0.50),
        "w_landing_gear":       (fsl * 0.40,         gl * 0.50),
        "w_cockpit_controls":   (ckl * 0.70,         gl + ckh * 0.40),
        "w_instruments":        (ckl * 0.50,         gl + ckh * 0.70),
        "w_furnishings":        (ckl + cbl * 0.50,   gl + cbh * 0.30),
        "w_air_conditioning":   (fsl * 0.50,         gl + 0.50),
        "w_propulsion_motors":  (fsl * 0.45,         gl + fsh + 0.50),
        "w_drive_system":       (fsl * 0.45,         gl + fsh * 0.80),
        "w_electrical":         (fsl * 0.45,         gl + 0.75),
        "w_flight_controls":    (ckl + cbl * 0.30,   gl + cbh * 0.50),
        "w_avionics":           (ckl * 0.60,         gl + ckh * 0.50),
        "w_structural_margin":  (fsl * 0.48,         gl + fsh * 0.45),
    }


def compute_cg(component_weights: dict,
               cg_locations: dict,
               extra_items: dict = None) -> tuple:
    """
    Compute aircraft CG given component weights + optional extra items.

    extra_items : {label: (weight_lbf_float, x_ft, z_ft)}

    Returns (x_cg_ft, z_cg_ft, total_weight_lbf).
    """
    sum_Wx = sum_Wz = sum_W = 0.0

    for name, wt in component_weights.items():
        if name in cg_locations:
            x, z = cg_locations[name]
            W = wt.to(lbf).magnitude
            sum_Wx += W * x
            sum_Wz += W * z
            sum_W  += W

    if extra_items:
        for _, (W, x, z) in extra_items.items():
            sum_Wx += W * x
            sum_Wz += W * z
            sum_W  += W

    if sum_W == 0:
        return 0.0, 0.0, 0.0
    return sum_Wx / sum_W, sum_Wz / sum_W, sum_W


def cg_loading_sequence(design: RotorcraftDesign,
                         component_weights: dict,
                         battery_weight: Q_) -> list:
    """
    Returns list of dicts with (label, x_cg, z_cg, W_total) for each loading step:
      Empty airframe → +Battery → +Electronics → +Defense → +Crew → +Passengers
    Also returns the sequence in reverse (unloading) for the full CG excursion.
    """
    f   = design.fuselage
    gl  = f.landing_gear_height.to(ft).magnitude
    ckl = f.cockpit_length.to(ft).magnitude
    cbl = f.cabin_length.to(ft).magnitude
    cbh = f.cabin_height.to(ft).magnitude
    ckh = f.cockpit_height.to(ft).magnitude

    W_batt = battery_weight.to(lbf).magnitude
    W_elec = design.payload.electronics.to(lbf).magnitude
    W_def  = design.payload.defense_system.to(lbf).magnitude
    W_pax  = design.payload.crew_and_passengers.to(lbf).magnitude
    w_pp   = W_pax / 10   # weight per person (2 crew + 8 pax assumed equal)

    # Positions for each payload item
    p_batt  = (ckl + cbl * 0.55,  gl + 0.50)           # battery: floor slab, aft-centre
    p_elec  = (ckl + 1.0,          gl + cbh * 0.50)     # electronics bay: fwd cabin
    p_def   = (ckl + cbl * 0.85,   gl + cbh * 0.30)     # defence: aft lower cabin
    p_crew  = (ckl * 0.50,          gl + ckh * 0.40)    # crew: cockpit
    p_pax   = (ckl + cbl * 0.50,   gl + cbh * 0.40)     # passengers: cabin mid

    cg_locs = component_cg_locations(design)

    steps = [
        ("Empty Airframe",  {}),
        ("+ Battery",       {"battery":     (W_batt,          *p_batt)}),
        ("+ Electronics",   {"battery":     (W_batt,          *p_batt),
                             "electronics": (W_elec,          *p_elec)}),
        ("+ Defense Sys",   {"battery":     (W_batt,          *p_batt),
                             "electronics": (W_elec,          *p_elec),
                             "defense":     (W_def,           *p_def)}),
        ("+ Crew (2)",      {"battery":     (W_batt,          *p_batt),
                             "electronics": (W_elec,          *p_elec),
                             "defense":     (W_def,           *p_def),
                             "crew":        (2 * w_pp,        *p_crew)}),
        ("+ Passengers (8)",{"battery":     (W_batt,          *p_batt),
                             "electronics": (W_elec,          *p_elec),
                             "defense":     (W_def,           *p_def),
                             "crew":        (2 * w_pp,        *p_crew),
                             "passengers":  (8 * w_pp,        *p_pax)}),
    ]

    results = []
    for label, extras in steps:
        x_cg, z_cg, W_tot = compute_cg(component_weights, cg_locs, extras)
        results.append({"label": label, "x_cg": x_cg,
                        "z_cg": z_cg, "W_total": W_tot})
    return results


# ============================================================
# Single-point analysis (all powers + energy for a given MTOW)
# ============================================================
def single_point_analysis(design: RotorcraftDesign,
                           mission: MissionProfile,
                           gross_weight: Q_,
                           alpha_deg: float = 10.0) -> dict:
    """
    Given an assumed gross weight (total aircraft weight in flight):
      1. Compute hover / climb / cruise / descent power at that weight.
      2. Compute mission energy from those powers.
      3. Compute battery weight from mission energy.
      4. Compute structural component weights at that gross weight.
      5. Return computed_mtow = W_empty + W_payload + W_battery.

    If computed_mtow == gross_weight the design closes.
    Uses ISA density: sea-level for hover/climb, cruise altitude for cruise.
    alpha_deg : cruise rotor tilt angle of attack (degrees).
    """
    W      = gross_weight.to(lbf)
    h_cr   = mission.cruise_altitude.to(ft).magnitude

    rho_sl = isa_density(0.0)   * slug / ft**3
    rho_cr = isa_density(h_cr)  * slug / ft**3

    # ---- Hover / takeoff (sea level) ----
    omega_hov  = to_omega(Q_(design.hover_rotor_speed_rpm, rpm))
    hov_cp_sl  = hover_power_coefficients(design, W, rho_sl, omega_hov)
    p_hov_sl   = hover_power_required(hov_cp_sl["C_P_hover"],
                                       rho_sl, design.rotor.radius, omega_hov)
    vi_sl      = hover_induced_velocity(W, rho_sl, design.rotor.radius)

    # ---- Axial climb (sea level density, conservative) ----
    climb_ax   = axial_power_required(p_hov_sl,
                                       mission.climb_rate.to(ft / s), vi_sl)
    p_climb    = climb_ax["P_climb_axial"]

    # ---- Cruise (forward flight at cruise altitude) ----
    V_cr      = mission.cruise_speed.to(ft / s)
    tip_cr    = tip_speed_schedule(V_cr)
    omega_cr  = omega_from_tip_speed(tip_cr, design.rotor.radius)
    alpha_cr  = np.deg2rad(alpha_deg) * rad
    T_cr      = thrust_from_weight_and_tilt(W, alpha_cr)

    fwd_cp    = forward_flight_power_coefficient(
        design=design, weight=W, T=T_cr, rho=rho_cr, omega=omega_cr,
        V_inf=V_cr, alpha=alpha_cr, climb_rate=0.0 * ft / s)
    p_cruise  = forward_flight_power_required(
        fwd_cp["C_P_forward_total"], rho_cr, design.rotor.radius, omega_cr)

    # ---- Descent (hover at cruise altitude, axial descent) ----
    hov_cp_cr = hover_power_coefficients(design, W, rho_cr, omega_hov)
    p_hov_cr  = hover_power_required(hov_cp_cr["C_P_hover"],
                                      rho_cr, design.rotor.radius, omega_hov)
    vi_cr     = hover_induced_velocity(W, rho_cr, design.rotor.radius)
    desc_ax   = axial_power_required(p_hov_cr,
                                      (-mission.descent_rate).to(ft / s), vi_cr)
    p_descent = (desc_ax["P_fast_descent_axial"]
                 if desc_ax["fast_descent_valid"]
                 else desc_ax["P_slow_descent_axial"])

    # ---- Mission energy + battery ----
    en   = mission_energy(mission, design.battery,
                          p_hover=p_hov_sl, p_climb=p_climb,
                          p_cruise=p_cruise, p_descent=p_descent)
    batt = battery_from_energy(en["E_nominal"], design.battery)

    # ---- Installed power (sized to hover with margin) ----
    p_installed = (p_hov_sl * design.aero.power_margin).to(hp)

    # ---- Component / structural weights ----
    comp_w  = component_weight_equations(
        design.rotor, W,
        design.fuselage.fuselage_length,
        design.fuselage.wetted_area_box_estimate,
        omega_hov)
    w_empty = sum_component_weights(comp_w)

    new_mtow = (w_empty + design.payload.total_payload + batt["weight"]).to(lbf)

    return {
        "p_hover_sl":      p_hov_sl,
        "p_hover_cr":      p_hov_cr,
        "p_climb":         p_climb,
        "p_cruise":        p_cruise,
        "p_descent":       p_descent,
        "p_installed":     p_installed,
        "mission_energy":  en,
        "battery":         batt,
        "component_weights": comp_w,
        "w_empty":         w_empty,
        "new_mtow":        new_mtow,
        "rho_sl":          rho_sl,
        "rho_cruise":      rho_cr,
        "omega_hover":     omega_hov,
        "omega_cruise":    omega_cr,
        "fwd_power_breakdown": fwd_cp,
        "hover_cp_sl":     hov_cp_sl,
    }


# ============================================================
# Weight closure (MTOW convergence loop)
# ============================================================
def weight_closure(design: RotorcraftDesign,
                   mission: MissionProfile,
                   max_iter: int = 60,
                   tol_lbf: float = 5.0) -> dict:
    """
    Iterates MTOW until:
      MTOW_new = W_empty(MTOW) + W_payload + W_battery(MTOW)
    converges to within tol_lbf pounds.
    """
    # gw = assumed gross weight fed into physics equations each iteration
    gw      = design.gw_guess.to(lbf)
    history = [gw.magnitude]

    converged = False
    for i in range(max_iter):
        result   = single_point_analysis(design, mission, gw)
        # computed_mtow = W_empty + W_payload + W_battery at this gross weight
        computed_mtow = result["new_mtow"]
        history.append(computed_mtow.magnitude)

        if abs((computed_mtow - gw).to(lbf).magnitude) < tol_lbf:
            gw        = computed_mtow
            converged = True
            break
        # Relaxed update: blend current gross weight toward computed MTOW
        gw = (0.5 * gw + 0.5 * computed_mtow).to(lbf)

    result["mtow_converged"]      = gw
    result["convergence_history"] = np.array(history)
    result["converged"]           = converged
    result["iterations"]          = i + 1
    return result


# ============================================================
# Performance sweeps
# ============================================================
def max_rate_of_climb_vs_altitude(design:         RotorcraftDesign,
                                   mtow:           Q_,
                                   p_installed_hp: float,
                                   altitudes_ft:   np.ndarray,
                                   v_min_kts:      float = 20,
                                   v_max_kts:      float = 170) -> tuple:
    """
    Max rate of climb (ft/min) at each altitude via excess-power method.

    ROC = (P_avail − P_req) × 550 / W × 60    [ft/min]

    All atmosphere data from standard_atmosphere.csv.
    P_avail lapsed with σ^0.12 (electric motor cooling derating).
    Sweeps forward speed 20–170 kts; best ROC = max excess power / W.
    Returns (roc_ft_per_min, v_best_kts) arrays.
    """
    V_sweep = np.linspace(v_min_kts, v_max_kts, 50)
    W       = mtow.to(lbf).magnitude
    roc_max = np.zeros(len(altitudes_ft))
    v_best  = np.zeros(len(altitudes_ft))

    for j, h in enumerate(altitudes_ft):
        rho        = csv_density(h) * slug / ft**3   # from CSV
        sigma      = csv_sigma(h)                     # from CSV
        p_avail_hp = p_installed_hp * sigma**0.12     # motor derating
        best_rc    = 0.0
        best_v     = 0.0

        for V_kts in V_sweep:
            V_fps  = (V_kts * kts).to(ft / s)
            tip    = tip_speed_schedule(V_fps)
            omega  = omega_from_tip_speed(tip, design.rotor.radius)
            alpha  = np.deg2rad(5.0) * rad           # shallow cruise-climb
            T      = thrust_from_weight_and_tilt(mtow, alpha)
            try:
                fcp = forward_flight_power_coefficient(
                    design=design, weight=mtow, T=T,
                    rho=rho, omega=omega, V_inf=V_fps,
                    alpha=alpha, climb_rate=0.0 * ft / s)
                p_req = forward_flight_power_required(
                    fcp["C_P_forward_total"], rho,
                    design.rotor.radius, omega).magnitude   # hp
                excess = p_avail_hp - p_req
                if excess > 0:
                    rc = excess * 550 / W * 60   # ft/min
                    if rc > best_rc:
                        best_rc = rc
                        best_v  = V_kts
            except Exception:
                pass

        roc_max[j] = best_rc
        v_best[j]  = best_v

    return roc_max, v_best


def max_speed_vs_altitude(design:         RotorcraftDesign,
                           mtow:           Q_,
                           p_installed_hp: float,
                           altitudes_ft:   np.ndarray,
                           M_adv_limit:    float = 0.78,
                           v_min_kts:      float = 60,
                           v_max_kts:      float = 215) -> np.ndarray:
    """
    Maximum level-flight speed (kts) at each altitude.

    ALL atmosphere data sourced from standard_atmosphere.csv:
      • density  → csv_density()          [kg/m³ → slug/ft³]
      • speed of sound → csv_speed_of_sound()  [m/s → ft/s]
      • density ratio σ → csv_sigma()     (for power lapse)

    Three constraints combine to make V_max decrease with altitude:

    1. Advancing-tip Mach limit (Fig 10.4 / Lecture 12 slide 15)
       M_adv = (V_tip + V_fwd) / a(h)
       a(h) drops with altitude → Mach constraint tightens.

    2. Power available lapse:
       Electric motors are air-cooled; cooling mass-flow ∝ ρ.
       P_avail(h) = P_installed × σ^0.12  (empirical eVTOL derating)
       This is much less than a turboshaft (σ^1.0) but still reduces
       available power.

    3. Power required change:
       At altitude ρ drops, so induced power (∝ T²/ρ at fixed μ)
       increases even though parasitic power decreases.

    Combined effect: V_max decreases monotonically with altitude,
    consistent with Lecture 12 slide 25.
    """
    v_max_arr = np.zeros(len(altitudes_ft))

    for j, h in enumerate(altitudes_ft):
        rho   = csv_density(h) * slug / ft**3           # from CSV
        a_fps = csv_speed_of_sound(h)                    # from CSV (m/s → ft/s)
        sigma = csv_sigma(h)                              # from CSV

        # Power available lapse: motor cooling degrades with thinner air
        p_avail_hp = p_installed_hp * sigma**0.12

        def power_excess(V_kts: float) -> float:
            V_fps_val = (V_kts * kts).to(ft / s)
            tip       = tip_speed_schedule(V_fps_val)
            omega     = omega_from_tip_speed(tip, design.rotor.radius)
            alpha     = np.deg2rad(10.0) * rad
            T         = thrust_from_weight_and_tilt(mtow, alpha)

            # Advancing-tip Mach constraint (Fig 10.4)
            V_tip_fps  = tip.to(ft / s).magnitude
            V_fwd_fps  = V_fps_val.to(ft / s).magnitude
            M_adv      = (V_tip_fps + V_fwd_fps) / a_fps
            if M_adv > M_adv_limit:
                return -1.0

            try:
                fcp   = forward_flight_power_coefficient(
                    design=design, weight=mtow, T=T,
                    rho=rho, omega=omega, V_inf=V_fps_val,
                    alpha=alpha, climb_rate=0.0 * ft / s)
                p_req = forward_flight_power_required(
                    fcp["C_P_forward_total"], rho,
                    design.rotor.radius, omega).magnitude
                return p_avail_hp - p_req
            except Exception:
                return -1.0

        V_sweep  = np.linspace(v_min_kts, v_max_kts, 300)
        feasible = [V for V in V_sweep if power_excess(V) >= 0]
        if feasible:
            v_max_arr[j] = max(feasible)
        else:
            v_max_arr[j] = 0.0

    return v_max_arr


# ============================================================
# Plotting helpers
# ============================================================
STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "text.color":       "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "lines.linewidth":  2,
    "font.size":        10,
}


def apply_style():
    plt.rcParams.update(STYLE)


ACCENT = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657", "#79c0ff"]


def plot_convergence(history: np.ndarray, ax=None):
    standalone = ax is None
    if standalone:
        apply_style()
        fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history, "o-", color=ACCENT[0], markersize=5)
    ax.axhline(history[-1], color=ACCENT[2], linestyle=":", linewidth=1.5,
               label=f"Converged: {history[-1]:,.0f} lb")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MTOW  (lb)")
    ax.set_title("MTOW Weight Closure Convergence")
    ax.legend()
    ax.grid(True)
    if standalone:
        plt.tight_layout()
        plt.show()


def plot_energy_timeline(en: dict, ax=None):
    """Stacked-area plot of energy remaining vs. mission time."""
    standalone = ax is None
    if standalone:
        apply_style()
        fig, ax = plt.subplots(figsize=(9, 4))

    # Build time array and power-per-phase
    phases = [
        ("Hover TO",  en["t_hover_takeoff"].to(minute).magnitude,  en["E_hover_takeoff"].to(kWh).magnitude),
        ("Climb",     en["t_climb"].to(minute).magnitude,           en["E_climb"].to(kWh).magnitude),
        ("Cruise",    en["t_cruise"].to(minute).magnitude,          en["E_cruise"].to(kWh).magnitude),
        ("Descent",   en["t_descent"].to(minute).magnitude,         en["E_descent"].to(kWh).magnitude),
        ("Hover Land",en["t_hover_landing"].to(minute).magnitude,   en["E_hover_landing"].to(kWh).magnitude),
    ]

    E_nom  = en["E_nominal"].to(kWh).magnitude
    t_cum  = [0.0]
    E_rem  = [E_nom]
    labels = []
    colors = []
    x_mid  = []

    for i, (label, dt, dE) in enumerate(phases):
        t_end = t_cum[-1] + dt
        E_end = E_rem[-1] - dE
        t_cum.append(t_end)
        E_rem.append(E_end)
        labels.append(label)
        colors.append(ACCENT[i % len(ACCENT)])
        x_mid.append((t_cum[-2] + t_cum[-1]) / 2)

    ax.plot(t_cum, E_rem, "o-", color=ACCENT[0])
    for i, (label, dt, dE) in enumerate(phases):
        t0, t1 = t_cum[i], t_cum[i+1]
        E0, E1 = E_rem[i], E_rem[i+1]
        ax.fill_between([t0, t1], [E0, E1], alpha=0.35, color=colors[i], label=label)
        ax.text(x_mid[i], (E0 + E1) / 2, f"{dE:.0f} kWh",
                ha="center", va="center", fontsize=8, color="white")

    ax.set_xlabel("Mission Time  (min)")
    ax.set_ylabel("Energy Remaining  (kWh)")
    ax.set_title("Energy Onboard — Max Range Mission (150 kts, 3 000 ft)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True)
    if standalone:
        plt.tight_layout()
        plt.show()


def plot_cg_excursion(cg_seq: list, fuselage_length_ft: float, ax=None):
    """CG x-position vs. loading step (and reversed unloading)."""
    standalone = ax is None
    if standalone:
        apply_style()
        fig, ax = plt.subplots(figsize=(8, 4))

    labels  = [s["label"]  for s in cg_seq]
    x_cg    = [s["x_cg"]   for s in cg_seq]
    W_tot   = [s["W_total"] for s in cg_seq]

    # Loading forward
    ax.plot(range(len(labels)), x_cg, "o-", color=ACCENT[0], label="Loading")
    # Unloading in reverse
    x_unload = x_cg[::-1]
    ax.plot(range(len(labels)), x_unload, "s--", color=ACCENT[2], label="Unloading")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Longitudinal CG  (ft from nose)")
    ax.set_title("CG Excursion — Loading / Unloading")
    ax.legend()
    ax.grid(True)

    # Reference lines at 40% and 60% of fuselage
    for frac, style in [(0.4, ":"), (0.6, ":")]:
        ax.axhline(frac * fuselage_length_ft, color=ACCENT[3],
                   linestyle=style, linewidth=1,
                   label=f"{int(frac*100)}% L_fus")
    if standalone:
        plt.tight_layout()
        plt.show()


def plot_roc_vs_altitude(altitudes_ft: np.ndarray,
                          roc: np.ndarray, ax=None):
    standalone = ax is None
    if standalone:
        apply_style()
        fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(roc, altitudes_ft / 1000, color=ACCENT[1])
    ax.set_xlabel("Max Rate of Climb  (ft/min)")
    ax.set_ylabel("Altitude  (kft)")
    ax.set_title("Max Rate of Climb vs. Altitude")
    ax.grid(True)
    ax.axvline(0, color=ACCENT[2], linewidth=1, linestyle="--",
               label="Service ceiling (ROC = 0)")
    ax.legend()
    if standalone:
        plt.tight_layout()
        plt.show()


def plot_speed_vs_altitude(altitudes_ft: np.ndarray,
                            v_max: np.ndarray, ax=None):
    standalone = ax is None
    if standalone:
        apply_style()
        fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(v_max, altitudes_ft / 1000, color=ACCENT[4])
    ax.axvline(170, color=ACCENT[2], linewidth=1, linestyle="--",
               label="170 kts requirement")
    ax.set_xlabel("Maximum Speed  (kts)")
    ax.set_ylabel("Altitude  (kft)")
    ax.set_title("Maximum Speed vs. Altitude")
    ax.legend()
    ax.grid(True)
    if standalone:
        plt.tight_layout()
        plt.show()


def plot_power_breakdown(fwd_cp: dict, p_hover_sl: Q_,
                          p_cruise: Q_, ax=None):
    """Bar chart comparing hover vs. cruise power components."""
    standalone = ax is None
    if standalone:
        apply_style()
        fig, ax = plt.subplots(figsize=(7, 4))

    labels = ["Hover\n(SL)", "Induced\n(cruise)", "Profile\n(cruise)",
              "Parasitic\n(cruise)", "Total\n(cruise)"]

    # Convert fractions of total cruise power to hp values for display
    p_cr_hp = p_cruise.magnitude
    p_h_hp  = p_hover_sl.magnitude
    vals    = [
        p_h_hp,
        fwd_cp["C_P_induced_fwd"]   / fwd_cp["C_P_forward_total"] * p_cr_hp,
        fwd_cp["C_P_profile_fwd"]   / fwd_cp["C_P_forward_total"] * p_cr_hp,
        fwd_cp["C_P_parasitic"]     / fwd_cp["C_P_forward_total"] * p_cr_hp,
        p_cr_hp,
    ]
    clrs = [ACCENT[0]] + [ACCENT[i+1] for i in range(3)] + [ACCENT[5]]
    bars = ax.bar(labels, vals, color=clrs, edgecolor="#30363d", linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f"{val:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Power  (hp)")
    ax.set_title("Hover vs. Cruise Power Breakdown")
    ax.grid(True, axis="y")
    if standalone:
        plt.tight_layout()
        plt.show()


def generate_all_plots(result: dict,
                        design: RotorcraftDesign,
                        cg_seq: list,
                        altitudes_ft: np.ndarray,
                        roc: np.ndarray,
                        v_max: np.ndarray):
    """
    Render each analysis as its own full-size report figure.
    Six separate windows — one per section — for clear readability.
    """
    apply_style()
    en   = result["mission_energy"]
    fus_len = design.fuselage.fuselage_length.to(ft).magnitude

    # ── Figure 1: Weight Closure ─────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig1.suptitle("Figure 1 — MTOW Weight Closure Iteration",
                  fontsize=14, color="#c9d1d9", y=0.98)
    history = result["convergence_history"]
    ax1.plot(history, "o-", color=ACCENT[0], markersize=7, linewidth=2)
    ax1.axhline(history[-1], color=ACCENT[2], linestyle=":", linewidth=1.5,
                label=f"Last value: {history[-1]:,.0f} lb")
    ax1.set_xlabel("Iteration  #", fontsize=12)
    ax1.set_ylabel("MTOW  (lb)", fontsize=12)
    ax1.set_title("Gross Weight → MTOW Feedback Loop\n"
                  "(divergence indicates infeasible design at current battery SE)",
                  fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(True)
    _annotate_converge(ax1, history)
    fig1.tight_layout()

    # ── Figure 2: Mission Energy Timeline ───────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(11, 6))
    fig2.suptitle("Figure 2 — Mission Energy Budget",
                  fontsize=14, color="#c9d1d9", y=0.98)
    plot_energy_timeline(en, ax=ax2)
    e_req = en["E_required"].to(kWh).magnitude
    e_nom = en["E_nominal"].to(kWh).magnitude
    ax2.set_title(f"Range mission: 250 nmi @ 150 kts, 3 000 ft\n"
                  f"Required = {e_req:.0f} kWh  |  Nominal (÷0.90) = {e_nom:.0f} kWh",
                  fontsize=10)
    fig2.tight_layout()

    # ── Figure 3: CG Excursion ───────────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(11, 6))
    fig3.suptitle("Figure 3 — Longitudinal CG Excursion",
                  fontsize=14, color="#c9d1d9", y=0.98)
    plot_cg_excursion(cg_seq, fus_len, ax=ax3)
    ax3.set_title(f"Fuselage length = {fus_len:.1f} ft  |  "
                  f"Loading sequence: empty → full crew & passengers",
                  fontsize=10)
    fig3.tight_layout()

    # ── Figure 4: ROC vs Altitude (CSV atmosphere) ──────────────────────────
    fig4, ax4 = plt.subplots(figsize=(9, 7))
    fig4.suptitle("Figure 4 — Rate of Climb vs. Altitude",
                  fontsize=14, color="#c9d1d9", y=0.98)
    ax4.plot(roc, altitudes_ft / 1000, color=ACCENT[1], linewidth=2)
    ax4.axvline(0,   color=ACCENT[2], linewidth=1.5, linestyle="--",
                label="Service ceiling  (ROC = 0 ft/min)")
    ax4.axhline(14,  color=ACCENT[3], linewidth=1,   linestyle=":",
                label="14 000 ft ceiling requirement")
    ax4.axvline(500, color=ACCENT[4], linewidth=1,   linestyle=":",
                label="500 ft/min ref climb rate")
    ax4.set_xlabel("Max Rate of Climb  (ft/min)", fontsize=12)
    ax4.set_ylabel("Altitude  (kft)", fontsize=12)
    # service ceiling annotation
    svc = altitudes_ft[roc > 0].max() / 1000 if np.any(roc > 0) else 0.0
    ax4.set_title(f"Density from standard_atmosphere.csv  |  "
                  f"Excess-power method:  ROC = (P_avail − P_req)×550 / W × 60\n"
                  f"Service ceiling ≈ {svc:.1f} kft",
                  fontsize=10)
    ax4.legend(fontsize=9)
    ax4.grid(True)
    fig4.tight_layout()

    # ── Figure 5: Max Speed vs Altitude (CSV atmosphere) ────────────────────
    fig5, ax5 = plt.subplots(figsize=(9, 7))
    fig5.suptitle("Figure 5 — Maximum Speed vs. Altitude",
                  fontsize=14, color="#c9d1d9", y=0.98)
    ax5.plot(v_max, altitudes_ft / 1000, color=ACCENT[4], linewidth=2)
    ax5.axvline(170, color=ACCENT[2], linewidth=1.5, linestyle="--",
                label="170 kts speed requirement")
    ax5.axhline(14,  color=ACCENT[3], linewidth=1,   linestyle=":",
                label="14 000 ft ceiling requirement")
    v_max_sl = v_max[0]
    ax5.set_xlabel("Maximum Level-Flight Speed  (kts)", fontsize=12)
    ax5.set_ylabel("Altitude  (kft)", fontsize=12)
    ax5.set_title(f"Density from standard_atmosphere.csv  |  "
                  f"Speed sweep 60–215 kts: highest V where P_req ≤ P_avail\n"
                  f"Advancing-tip Mach ≤ 0.78 constraint applied  |  "
                  f"V_max at SL ≈ {v_max_sl:.0f} kts",
                  fontsize=10)
    ax5.legend(fontsize=9)
    ax5.grid(True)
    fig5.tight_layout()

    # ── Figure 6: Power Breakdown ────────────────────────────────────────────
    fig6, ax6 = plt.subplots(figsize=(9, 6))
    fig6.suptitle("Figure 6 — Power Budget: Hover vs. Cruise Components",
                  fontsize=14, color="#c9d1d9", y=0.98)
    plot_power_breakdown(result["fwd_power_breakdown"],
                         result["p_hover_sl"], result["p_cruise"], ax=ax6)
    ax6.set_title("Momentum + blade element theory — components at design point",
                  fontsize=10)
    fig6.tight_layout()

    # ── Interactive cursors (click/hover to see data values) ────────────────
    try:
        import mplcursors
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            for ax in fig.axes:
                for line in ax.get_lines():
                    if len(line.get_xdata()) > 1:
                        mplcursors.cursor(line, hover=True)
    except ImportError:
        pass   # mplcursors not installed — plots still work, just no hover

    plt.show()


def _annotate_converge(ax, history):
    """Add a text note on the convergence plot."""
    ax.annotate(f"Start: {history[0]:,.0f} lb",
                xy=(0, history[0]), xytext=(0.5, history[0]),
                fontsize=8, color="#8b949e")


# ============================================================
# Summary printer
# ============================================================
def print_design_summary(result: dict, design: RotorcraftDesign,
                          cg_seq: list, roc: np.ndarray,
                          v_max: np.ndarray, altitudes_ft: np.ndarray):
    en   = result["mission_energy"]
    batt = result["battery"]
    cw   = result["component_weights"]
    mtow = result["mtow_converged"].to(lbf).magnitude
    w_e  = result["w_empty"].to(lbf).magnitude
    w_b  = batt["weight"].to(lbf).magnitude
    w_p  = design.payload.total_payload.to(lbf).magnitude

    # Service ceiling = highest altitude where ROC > 0
    svc_ceil = altitudes_ft[roc > 0].max() if np.any(roc > 0) else 0.0
    # Altitude where Vmax >= 170 kts
    v170_alts = altitudes_ft[v_max >= 170]
    max_alt_170 = v170_alts.max() if len(v170_alts) > 0 else 0.0

    W  = "═" * 55
    w  = "─" * 55
    def row(label, value, unit=""):
        return f"  {label:<34} {value:>12}  {unit}"

    print(f"\n╔{W}╗")
    print(f"║{'   MARINE ONE eVTOL — DESIGN SUMMARY':^55}║")
    print(f"╠{W}╣")
    print(f"║  {'WEIGHTS':}{'':47}║")
    print(f"║{w}║")
    print(f"║{row('MTOW (converged)',        f'{mtow:,.0f}',    'lbf')}║")
    print(f"║{row('Empty weight',            f'{w_e:,.0f}',     'lbf')}║")
    print(f"║{row('Battery weight',          f'{w_b:,.0f}',     'lbf')}║")
    print(f"║{row('Payload weight',          f'{w_p:,.0f}',     'lbf')}║")
    print(f"║{row('Empty-weight fraction',   f'{w_e/mtow:.3f}', '')}║")
    print(f"║{row('Battery weight fraction', f'{w_b/mtow:.3f}', '')}║")
    # Pre-compute all formatted strings to avoid nested f-string quoting issues
    p_hov   = f"{result['p_hover_sl'].magnitude:,.0f}"
    p_inst  = f"{result['p_installed'].magnitude:,.0f}"
    p_cl    = f"{result['p_climb'].magnitude:,.0f}"
    p_cr    = f"{result['p_cruise'].magnitude:,.0f}"
    p_des   = f"{result['p_descent'].magnitude:,.0f}"

    e_hto   = f"{en['E_hover_takeoff'].to(kWh).magnitude:.1f}"
    e_clb   = f"{en['E_climb'].to(kWh).magnitude:.1f}"
    e_cru   = f"{en['E_cruise'].to(kWh).magnitude:.1f}"
    e_des   = f"{en['E_descent'].to(kWh).magnitude:.1f}"
    e_hld   = f"{en['E_hover_landing'].to(kWh).magnitude:.1f}"
    e_req   = f"{en['E_required'].to(kWh).magnitude:.1f}"
    e_nom   = f"{en['E_nominal'].to(kWh).magnitude:.1f}"
    b_wt    = f"{batt['weight'].to(lbf).magnitude:,.0f}"

    r_rad   = f"{design.rotor.radius.to(ft).magnitude:.1f}"
    r_sol   = f"{solidity(design.rotor.n_blades, design.rotor.chord, design.rotor.radius):.4f}"
    _omega_rads = result['omega_hover'].to(1 / ureg.s).magnitude
    r_omg   = f"{_omega_rads:.2f}"
    r_mu    = f"{result['fwd_power_breakdown']['mu']:.4f}"
    r_ct    = f"{result['fwd_power_breakdown']['C_T']:.6f}"

    s_ceil  = f"{svc_ceil / 1000:.1f}"
    s_v170  = f"{max_alt_170 / 1000:.1f}"
    s_roc0  = f"{roc[0]:,.0f}"

    x_vals  = [s['x_cg'] for s in cg_seq]
    cg_rng  = f"{min(x_vals):.2f} - {max(x_vals):.2f}"

    print(f"╠{W}╣")
    print(f"║  {'POWER':}{'':49}║")
    print(f"║{w}║")
    print(f"║{row('Hover power (SL)',        p_hov,  'hp')}║")
    print(f"║{row('Installed power',         p_inst, 'hp')}║")
    print(f"║{row('Climb power (axial)',      p_cl,   'hp')}║")
    print(f"║{row('Cruise power (3 000 ft)', p_cr,   'hp')}║")
    print(f"║{row('Descent power',           p_des,  'hp')}║")
    print(f"╠{W}╣")
    print(f"║  {'MISSION ENERGY':}{'':43}║")
    print(f"║{w}║")
    print(f"║{row('Hover TO energy',        e_hto,  'kWh')}║")
    print(f"║{row('Climb energy',           e_clb,  'kWh')}║")
    print(f"║{row('Cruise energy',          e_cru,  'kWh')}║")
    print(f"║{row('Descent energy',         e_des,  'kWh')}║")
    print(f"║{row('Hover land energy',      e_hld,  'kWh')}║")
    print(f"║{row('Total required energy',  e_req,  'kWh')}║")
    print(f"║{row('Nominal battery energy', e_nom,  'kWh')}║")
    print(f"║{row('Battery weight',          b_wt,   'lb')}║")
    print(f"╠{W}╣")
    print(f"║  {'ROTOR & AERODYNAMICS':}{'':37}║")
    print(f"║{w}║")
    print(f"║{row('Rotor radius',           r_rad, 'ft')}║")
    print(f"║{row('Rotor solidity',         r_sol, '')}║")
    print(f"║{row('Hover omega',            r_omg, 'rad/s')}║")
    print(f"║{row('Cruise advance ratio u', r_mu,  '')}║")
    print(f"║{row('Cruise CT',              r_ct,  '')}║")
    print(f"╠{W}╣")
    print(f"║  {'PERFORMANCE':}{'':46}║")
    print(f"║{w}║")
    print(f"║{row('Service ceiling (ROC=0)', s_ceil,  'kft')}║")
    print(f"║{row('Max alt at 170 kts',      s_v170,  'kft')}║")
    print(f"║{row('Max ROC at sea level',    s_roc0,  'ft/min')}║")
    print(f"╠{W}╣")
    print(f"║  {'CG EXCURSION (longitudinal)':}{'':30}║")
    print(f"║{w}║")
    for s in cg_seq:
        xcg_str = f"{s['x_cg']:.2f}"
        print(f"║{row(s['label'], xcg_str, 'ft from nose')}║")
    print(f"║{row('CG range (fwd-aft)', cg_rng, 'ft')}║")
    print(f"╠{W}╣")
    print(f"║  {'COMPONENT WEIGHTS (empty)':}{'':32}║")
    print(f"║{w}║")
    for name, val in cw.items():
        short = name.replace("w_", "").replace("_", " ").title()
        print(f"║{row(short, f'{val.to(lbf).magnitude:,.0f}', 'lbf')}║")
    print(f"║{row('TOTAL EMPTY', f'{w_e:,.0f}', 'lbf')}║")
    print(f"╚{W}╝\n")


# ============================================================
# Design setup
# ============================================================
design = RotorcraftDesign(
    rotor=RotorGeometry(
        radius=19.5 * ft,        # 39 ft diameter — fits the 40×40 ft pad
        chord=2.0  * ft,
        n_blades=4,
        n_wheels=3,
    ),
    fuselage=FuselageGeometry(
        cockpit_length=10 * ft,
        cockpit_width=8  * ft,
        cockpit_height=6 * ft,
        cabin_length=16  * ft,
        cabin_width=8    * ft,
        cabin_height=6   * ft,
        landing_gear_height=3.0 * ft,
    ),
    payload=PayloadWeights(
        crew_and_passengers=2_000 * lbf,   # 10 people × 200 lb
        electronics=1_000         * lbf,
        defense_system=1_000      * lbf,
    ),
    aero=AeroCoefficients(
        cd0_hover=0.0010,
        induced_power_factor=1.15,
        profile_power_factor=4.7,
        power_margin=1.25,             # 25 % margin over hover power
    ),
    battery=BatteryModel(
        specific_energy=0.300 * kWh / kg,
        usable_fraction=0.90,
    ),
    gw_guess=25_000 * lbf,
)

mission = MissionProfile(
    cruise_range=250       * nmi,
    cruise_altitude=3_000  * ft,
    climb_rate=500         * ft / minute,
    cruise_speed=150       * kts,
    descent_rate=500       * ft / minute,
    hover_time_takeoff=2   * minute,
    hover_time_landing=2   * minute,
)

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Step 1 — single-pass at the initial MTOW guess (White Hawk basis).
    #   Power → Energy → Battery weight.  This is always the reference
    #   for plots and performance sweeps so they are finite/meaningful.
    # ------------------------------------------------------------------
    print("\n  Single-pass analysis at assumed gross weight …")
    gw = design.gw_guess.to(lbf)                          # 25,000 lbf — fixed input
    ref = single_point_analysis(design, mission, gw)
    computed_mtow = ref["new_mtow"]                        # W_empty + W_payload + W_battery

    w_batt_lbf    = ref["battery"]["weight"].to(lbf).magnitude
    w_empty_lbf   = ref["w_empty"].to(lbf).magnitude
    w_payload_lbf = design.payload.total_payload.to(lbf).magnitude

    print(f"  Assumed gross weight : {gw.magnitude:,.0f} lb")
    print(f"  Battery required     : {w_batt_lbf:,.0f} lb")
    print(f"  Empty weight         : {w_empty_lbf:,.0f} lb")
    print(f"  Payload              : {w_payload_lbf:,.0f} lb")
    print(f"  Computed MTOW        : {computed_mtow.to(lbf).magnitude:,.0f} lb  "
          f"({'CLOSES' if abs(computed_mtow.to(lbf).magnitude - gw.magnitude) < 500 else 'DOES NOT CLOSE — design infeasible at 300 Wh/kg / 250 nmi'})")

    # ------------------------------------------------------------------
    # Step 2 — weight closure loop (12 iters — enough to show divergence)
    # ------------------------------------------------------------------
    print("\n  Running weight closure loop …")
    closure = weight_closure(design, mission, max_iter=12, tol_lbf=5.0)

    if closure["converged"]:
        print(f"  Converged in {closure['iterations']} iterations "
              f"→ MTOW = {closure['mtow_converged'].to(lbf).magnitude:,.0f} lb")
    else:
        last = closure["convergence_history"][-1]
        print(f"  Did NOT converge (battery weight > gross weight).")
        print(f"  This is expected: 300 Wh/kg + 250 nmi is at the limit")
        print(f"  of feasibility for this vehicle class.")
        print(f"  Last iterated MTOW: {last:,.0f} lb  (diverging — see convergence plot)")

    # All CG / performance / plots use the design gross weight (gw)
    # so numbers remain physically meaningful regardless of closure.

    # ---- CG excursion ----
    cg_seq = cg_loading_sequence(
        design,
        ref["component_weights"],
        ref["battery"]["weight"])

    # ---- Performance sweeps at design gross weight (gw_guess = 25,000 lbf) ----
    altitudes  = np.linspace(0, 14_000, 70)
    p_inst_hp  = ref["p_installed"].magnitude

    print("\n  Computing max ROC vs altitude …")
    roc, v_best = max_rate_of_climb_vs_altitude(
        design, gw, p_inst_hp, altitudes)

    print("  Computing max speed vs altitude …")
    v_max = max_speed_vs_altitude(
        design, gw, p_inst_hp, altitudes)

    # Attach closure history; label converged value correctly
    ref["convergence_history"] = closure["convergence_history"]
    ref["mtow_converged"]      = closure["mtow_converged"] if closure["converged"] else computed_mtow
    result = ref

    # ---- Summary ----
    print_design_summary(result, design, cg_seq, roc, v_max, altitudes)

    # ---- Plots ----
    generate_all_plots(result, design, cg_seq, altitudes, roc, v_max)
