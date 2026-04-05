"""
interactive_design.py  --  Marine One eVTOL Interactive Design Tool
=================================================================
Uses project_2_revised.py as the computation engine.

INPUT SECTIONS
--------------
  Design Characteristics:
    GW guess, rotor geometry, fuselage LWH, N_people,
    K_electronics, K_battery  (volume-density coefficients)
  Performance Characteristics:
    battery SE, cruise speed, AoA, climb/descent rates, rotor RPM,
    mission profile (range / cruise alt / hover time),
    payload breakdown (crew / electronics / defense)
  Sweep Parameters & Tip Speed Constraints:
    ROC sweep V range, Vmax sweep V range, altitude sweep range/points,
    noise limit, SKE limit, compressibility Mach, max advance ratio mu

OUTPUT (separate resizable window)
-----------------------------------
  1. Mission energy breakdown
  2. Battery weight
  3. Weight statement + feasibility (DW)
  4. Min battery SE required (with math)
  5. Fuselage volume sizing (Vmargin formula)

PLOTS
-----
  Six full-size matplotlib windows (separate from UI).
  Fig 10.4 tip-speed chart opens via dedicated button.
"""

import sys
import os
sys.stdout.reconfigure(encoding="utf-8")

import json
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import numpy as np
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# -- engine import -----------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from project_2_revised import (
    ureg, Q_, ft, s, lbf, kts, nmi, hp, kg, kWh, minute, slug, rad, g_ft,
    RotorGeometry, FuselageGeometry, PayloadWeights, AeroCoefficients,
    BatteryModel, MissionProfile, RotorcraftDesign,
    single_point_analysis,
    max_rate_of_climb_vs_altitude, max_speed_vs_altitude,
    cg_loading_sequence, generate_all_plots,
    fuselage_volume_check,
    tip_speed_schedule,
    apply_style, ACCENT,
)

# ---------------------------------------------------------------------------
# UI defaults  (all editable at runtime)
# ---------------------------------------------------------------------------
_D_GW          = 25000
_D_RADIUS      = 19.5
_D_BLADES      = 4
_D_WHEELS      = 3
_D_CHORD       = 2.0
_D_CK_L, _D_CK_W, _D_CK_H = 10, 8, 6
_D_CA_L, _D_CA_W, _D_CA_H = 16, 8, 6
_D_NPEOP       = 10
_D_K_ELEC      = 0.020
_D_K_BATT      = 0.025

_D_SE          = 300
_D_VFWD        = 150
_D_AOA         = 10
_D_CLIMB       = 500
_D_DESCENT     = 500
_D_RPM         = 300

_D_RANGE       = 250
_D_ALT         = 3_000
_D_HOVER_T     = 2.0
_D_CREW        = 2_000
_D_ELEC        = 1_000
_D_DEF         = 1_000

_D_ROC_VMIN    = 20
_D_ROC_VMAX    = 170
_D_VMAX_VMIN   = 60
_D_VMAX_VMAX   = 215
_D_ALT_MAX     = 14_000
_D_ALT_PTS     = 50

_D_TIP_NOISE   = 780      # ft/s -- noise limit
_D_TIP_SKE     = 390      # ft/s -- stored kinetic energy limit
_D_TIP_M_COMP  = 0.92     # Mach -- compressibility onset (for chart display)
_D_TIP_MU_MAX  = 0.45     # --   -- retreating blade stall advance ratio limit

_A_SL = 1116.4            # ft/s, speed of sound at sea level (ISA standard)


# ---------------------------------------------------------------------------
# Helper parsers
# ---------------------------------------------------------------------------
def _fval(v, d):
    try:    return float(v.get())
    except: return d

def _ival(v, d):
    try:    return int(v.get())
    except: return d


# ---------------------------------------------------------------------------
# Core analysis runner
# ---------------------------------------------------------------------------
def run_analysis(inp: dict) -> dict:
    design = RotorcraftDesign(
        rotor=RotorGeometry(
            radius=inp["rotor_radius"] * ft,
            chord=inp["chord"] * ft,
            n_blades=inp["n_blades"],
            n_wheels=inp["n_wheels"]),
        fuselage=FuselageGeometry(
            cockpit_length=inp["ck_l"] * ft, cockpit_width=inp["ck_w"] * ft,
            cockpit_height=inp["ck_h"] * ft, cabin_length=inp["ca_l"] * ft,
            cabin_width=inp["ca_w"] * ft, cabin_height=inp["ca_h"] * ft,
            landing_gear_height=3.0 * ft),
        payload=PayloadWeights(
            crew_and_passengers=inp["payload_crew"] * lbf,
            electronics=inp["payload_elec"] * lbf,
            defense_system=inp["payload_def"] * lbf),
        aero=AeroCoefficients(
            cd0_hover=0.001, induced_power_factor=1.15,
            profile_power_factor=4.7, power_margin=1.25),
        battery=BatteryModel(
            specific_energy=(inp["battery_se"] / 1000.0) * kWh / kg,
            usable_fraction=0.90),
        gw_guess=inp["gw_guess"] * lbf,
        hover_rotor_speed_rpm=inp["rotor_rpm"],
    )
    mission = MissionProfile(
        cruise_range=inp["range_nmi"] * nmi,
        cruise_altitude=inp["cruise_alt_ft"] * ft,
        climb_rate=inp["climb_rate"] * ft / minute,
        cruise_speed=inp["cruise_speed"] * kts,
        descent_rate=inp["descent_rate"] * ft / minute,
        hover_time_takeoff=inp["hover_time_min"] * minute,
        hover_time_landing=inp["hover_time_min"] * minute)

    gw  = design.gw_guess.to(lbf)
    ref = single_point_analysis(design, mission, gw, alpha_deg=inp["aoa_deg"])

    n_pts    = max(2, inp["alt_pts"])
    alt_max  = max(1000, inp["alt_max"])
    altitudes = np.linspace(0, alt_max, n_pts)

    p_inst_hp = ref["p_installed"].magnitude

    roc, v_best = max_rate_of_climb_vs_altitude(
        design, gw, p_inst_hp, altitudes,
        v_min_kts=inp["roc_vmin"], v_max_kts=inp["roc_vmax"])

    v_max = max_speed_vs_altitude(
        design, gw, p_inst_hp, altitudes,
        v_min_kts=inp["vmax_vmin"], v_max_kts=inp["vmax_vmax"])

    vol = fuselage_volume_check(
        design, ref["battery"]["weight"],
        n_occupants=inp["n_people"],
        k_electronics_ft3_lb=inp["k_elec"],
        k_battery_ft3_lb=inp["k_batt"])

    cg_seq = cg_loading_sequence(
        design, ref["component_weights"], ref["battery"]["weight"])

    return {
        "design": design, "mission": mission, "ref": ref, "gw": gw,
        "altitudes": altitudes, "roc": roc, "v_max": v_max,
        "cg_seq": cg_seq, "vol": vol,
        "k_elec": inp["k_elec"], "k_batt": inp["k_batt"],
        "n_people": inp["n_people"],
    }


# ---------------------------------------------------------------------------
# Output formatter
# ---------------------------------------------------------------------------
def format_output(r: dict) -> str:
    ref    = r["ref"]
    design = r["design"]
    mission = r["mission"]
    gw     = r["gw"]
    en     = ref["mission_energy"]
    batt   = ref["battery"]
    vol    = r["vol"]

    gw_lb      = gw.magnitude
    w_empty_lb = ref["w_empty"].to(lbf).magnitude
    w_batt_lb  = batt["weight"].to(lbf).magnitude
    w_pay_lb   = design.payload.total_payload.to(lbf).magnitude
    mtow_lb    = ref["new_mtow"].to(lbf).magnitude
    se_wh_kg   = design.battery.specific_energy.to(kWh / kg).magnitude * 1000.0
    e_nom_kWh  = en["E_nominal"].to(kWh).magnitude
    k_elec     = r["k_elec"]
    k_batt     = r["k_batt"]
    n_ppl      = r["n_people"]

    # Payload breakdown from design object (reflects user inputs)
    w_crew_lb = design.payload.crew_and_passengers.to(lbf).magnitude
    w_elec_lb = design.payload.electronics.to(lbf).magnitude
    w_def_lb  = design.payload.defense_system.to(lbf).magnitude

    # Mission values from mission object (reflects user inputs)
    rng_nmi   = mission.cruise_range.to(nmi).magnitude
    alt_ft    = mission.cruise_altitude.to(ft).magnitude
    vfwd_kts  = mission.cruise_speed.to(kts).magnitude
    hov_min   = mission.hover_time_takeoff.to(minute).magnitude

    dw = gw_lb - (w_empty_lb + w_pay_lb + w_batt_lb)
    w_batt_budget_lb = gw_lb - w_empty_lb - w_pay_lb
    m_batt_budget_sl = w_batt_budget_lb / 32.174
    m_batt_budget_kg = m_batt_budget_sl * 14.5939
    se_min_wh_kg = (e_nom_kWh * 1000.0) / m_batt_budget_kg if m_batt_budget_kg > 0 else float("inf")
    ratio = se_min_wh_kg / se_wh_kg if se_wh_kg > 0 else float("inf")

    if   0 <= dw < 200:  fs, fy = "CLOSES  (DW \u2248 0)",                    "[OK]"
    elif dw >= 200:      fs, fy = "GW GUESS TOO HIGH -- may reduce",        "[^]"
    elif dw >= -200:     fs, fy = "BARELY MISSES -- increase GW slightly",  "[~]"
    else:                fs, fy = "DOES NOT CLOSE -- GW too low",           "[X]"

    S  = "=" * 66
    s2 = "-" * 66
    L  = []
    a  = L.append

    # 1. Mission params used
    a(f"\n{S}")
    a("  MISSION PARAMETERS USED")
    a(s2)
    a(f"  Range          : {rng_nmi:.0f} nmi")
    a(f"  Cruise alt     : {alt_ft:,.0f} ft")
    a(f"  Cruise speed   : {vfwd_kts:.0f} kts")
    a(f"  Hover time     : {hov_min:.1f} min each end")
    a(f"  Payload        : {w_crew_lb:,.0f} lb crew/pax  +  "
      f"{w_elec_lb:,.0f} lb electronics  +  {w_def_lb:,.0f} lb defense")

    # 2. Mission energy
    a(f"\n{S}")
    a("  SECTION 1 -- MISSION ENERGY BREAKDOWN")
    a(s2)
    a(f"  {'Phase':<22} {'Time (min)':>10}  {'Energy (kWh)':>12}")
    a(s2)
    for nm, t, e in [
        ("Hover Takeoff",  en["t_hover_takeoff"],  en["E_hover_takeoff"]),
        ("Climb",          en["t_climb"],           en["E_climb"]),
        ("Cruise",         en["t_cruise"],          en["E_cruise"]),
        ("Descent",        en["t_descent"],         en["E_descent"]),
        ("Hover Landing",  en["t_hover_landing"],   en["E_hover_landing"]),
    ]:
        a(f"  {nm:<22} {t.to(minute).magnitude:>10.1f}  {e.to(kWh).magnitude:>12.1f}")
    a(s2)
    a(f"  {'Total Required':<22} {'':>10}  {en['E_required'].to(kWh).magnitude:>12.1f}")
    a(f"  {'Nominal (/ 0.90)':<22} {'':>10}  {e_nom_kWh:>12.1f}")

    # 3. Battery
    a(f"\n{S}")
    a("  SECTION 2 -- BATTERY")
    a(s2)
    a(f"  Specific energy (input)  : {se_wh_kg:,.0f} Wh/kg")
    a(f"  Usable fraction          : 90 %")
    a(f"  Nominal energy required  : {e_nom_kWh:,.1f} kWh")
    a(f"  Battery weight           : {w_batt_lb:,.0f} lb")

    # 4. Weight statement
    a(f"\n{S}")
    a("  SECTION 3 -- WEIGHT STATEMENT")
    a(s2)
    a(f"  {'Item':<42} {'Weight (lb)':>10}")
    a(s2)
    a(f"  {'Assumed Gross Weight (GW_guess) [INPUT]':<42} {gw_lb:>10,.0f}")
    a(s2)
    a(f"  {'Empty Weight            (computed)':<42} {w_empty_lb:>10,.0f}")
    a(f"  {'Payload Weight          (user input)':<42} {w_pay_lb:>10,.0f}")
    a(f"    {'  Crew & Passengers':<40} {w_crew_lb:>10,.0f}")
    a(f"    {'  Electronics Bay':<40} {w_elec_lb:>10,.0f}")
    a(f"    {'  Defensive Armament':<40} {w_def_lb:>10,.0f}")
    a(f"  {'Battery Weight          (computed)':<42} {w_batt_lb:>10,.0f}")
    a(s2)
    a(f"  {'Computed MTOW = W_empty+W_pay+W_batt':<42} {mtow_lb:>10,.0f}")

    # 5. Feasibility
    a(f"\n{S}")
    a("  SECTION 4 -- FEASIBILITY CHECK")
    a(s2)
    a(f"  DW = GW_guess - (W_empty + W_payload + W_battery)")
    a(f"     = {gw_lb:,.0f} - ({w_empty_lb:,.0f} + {w_pay_lb:,.0f} + {w_batt_lb:,.0f})")
    a(f"     = {gw_lb:,.0f} - {w_empty_lb + w_pay_lb + w_batt_lb:,.0f}")
    a(f"     = {dw:+,.0f} lb")
    a("")
    a(f"  {fy}  {fs}")
    if 0 <= dw < 200:
        a("")
        a(f"  Computed MTOW ({mtow_lb:,.0f} lb) is within {dw:,.0f} lb of GW guess. Design closes.")
    elif dw >= 200:
        a("")
        a(f"  GW_guess ({gw_lb:,.0f} lb) is conservative by {dw:,.0f} lb.")
        a(f"  Computed MTOW = {mtow_lb:,.0f} lb. You may lower GW guess.")
    elif dw >= -200:
        a("")
        a(f"  Computed MTOW ({mtow_lb:,.0f} lb) exceeds GW guess by {abs(dw):,.0f} lb.")
        a(f"  Increase GW guess by at least {abs(dw):,.0f} lb and re-run.")
    else:
        a("")
        a(f"  Computed MTOW ({mtow_lb:,.0f} lb) exceeds GW guess ({gw_lb:,.0f} lb) by {abs(dw):,.0f} lb.")
        a("  Increase GW, reduce range/payload, or raise battery specific energy.")

    # 6. Minimum SE
    a(f"\n{S}")
    a("  SECTION 5 -- MINIMUM BATTERY SPECIFIC ENERGY")
    a(s2)
    a(f"  Step 1: W_budget = GW - W_empty - W_payload = {w_batt_budget_lb:,.0f} lb")
    a(f"  Step 2: m_budget = {w_batt_budget_lb:,.0f}/32.174 = "
      f"{m_batt_budget_sl:,.1f} sl = {m_batt_budget_kg:,.1f} kg")
    a(f"  Step 3: E_nominal = {e_nom_kWh:,.1f} kWh")
    if m_batt_budget_kg > 0:
        a(f"  Step 4: SE_min = {e_nom_kWh:,.1f} kWh / {m_batt_budget_kg:,.1f} kg "
          f"= {se_min_wh_kg:,.0f} Wh/kg")
        a(f"  Current SE = {se_wh_kg:,.0f} Wh/kg  |  Ratio = {ratio:.2f}x")
        if se_min_wh_kg <= se_wh_kg:
            a("  [OK] Current SE satisfies the mission.")
        else:
            a(f"  [X] Need {ratio:.2f}x more energy density.")
    else:
        a("  [!] Battery weight budget <= 0 -- GW guess too low.")

    # 7. Volume sizing
    a(f"\n{S}")
    a("  SECTION 6 -- FUSELAGE VOLUME SIZING CHECK")
    a(s2)
    a("  Formulas (from reference):")
    a("    V_usable  = Cockpit_Volume + Cabin_Volume")
    a("    V_payload = V_seats_tot + V_electronics")
    a(f"    V_seats   = N_people * (L_seat * W_seat * H_seat)  [2.5 x 1.5 x 3.0 ft]")
    a(f"    V_elec    = K_elec * W_electronics")
    a(f"    V_batt    = K_batt * W_battery")
    a("    V_margin  = V_usable - (V_batt + V_payload)")
    a(s2)
    a(f"  User inputs:")
    a(f"    N_people      = {n_ppl}")
    a(f"    K_electronics = {k_elec:.4f} ft^3/lb")
    a(f"    K_battery     = {k_batt:.4f} ft^3/lb")
    a(s2)
    v = r["vol"]
    a(f"  {'Cockpit Volume (Lc*Wc*Hc)':<42} {v['V_cockpit_ft3']:>10.1f} ft^3")
    a(f"  {'Cabin Volume (Lcab*Wcab*Hcab)':<42} {v['V_cabin_ft3']:>10.1f} ft^3")
    a(f"  {'V_usable (total interior)':<42} {v['V_usable_ft3']:>10.1f} ft^3")
    a(s2)
    a(f"  {'V_seats = ' + str(n_ppl) + '*(2.5*1.5*3.0)':<42} {v['V_seats_ft3']:>10.1f} ft^3")
    a(f"  {'V_electronics = K*W_elec':<42} {v['V_electronics_ft3']:>10.1f} ft^3")
    a(f"  {'V_payload (seats+electronics)':<42} {v['V_payload_ft3']:>10.1f} ft^3")
    a(f"  {'V_battery = K_batt*W_batt':<42} {v['V_battery_ft3']:>10.1f} ft^3")
    a(s2)
    a(f"  {'V_margin = V_usable-(V_batt+V_pay)':<42} {v['V_margin_ft3']:>10.1f} ft^3")
    a("")
    if v["fits"]:
        a("  [OK] V_margin >= 0 -- components FIT the fuselage.")
    else:
        a(f"  [X] V_margin < 0 -- fuselage too small by {abs(v['V_margin_ft3']):.1f} ft^3.")
        a("       Increase cabin/cockpit dims or adjust K coefficients.")
    a(f"\n{S}")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Prouty Figure 10.4  --  Tip Speed Constraints Chart
# ---------------------------------------------------------------------------
def show_tip_speed_chart(inp: dict, last_result=None):
    """
    Recreate Prouty Figure 10.4 in matplotlib.

    Four constraint regions are shown:
      Noise Limit       -- max allowable tip speed (acoustics)
      Compressibility   -- advancing-tip Mach onset: V_tip + V_fwd = M_comp * a_sl
      Stall Limit       -- retreating blade stall: V_tip = V_fwd / mu_max
      SKE Limit         -- minimum tip speed for stored kinetic energy on engine failure

    The acceptable operating region lies between all four limits.
    Your design's hover tip speed (from RPM x radius) is overlaid,
    as is the engine tip-speed schedule used in the performance sweeps.
    """
    noise  = inp["tip_noise"]       # ft/s
    ske    = inp["tip_ske"]         # ft/s
    M_comp = inp["tip_M_comp"]      # Mach
    mu_max = inp["tip_mu_max"]      # dimensionless

    # Hover tip speed from the user's RPM and rotor radius inputs
    omega_hover   = inp["rotor_rpm"] * 2.0 * np.pi / 60.0   # rad/s
    V_tip_hover   = omega_hover * inp["rotor_radius"]         # ft/s

    V_kts_arr = np.linspace(0, 220, 600)
    V_fps_arr = V_kts_arr * 1.68781                           # ft/s

    # Compressibility limit: V_tip + V_fwd = M_comp * a_sl
    V_tip_comp  = M_comp * _A_SL - V_fps_arr

    # Stall limit: V_tip = V_fwd / mu_max  (retreating blade stall)
    V_tip_stall = V_fps_arr / mu_max

    # Tip-speed schedule from the engine (what the sweeps actually use)
    sched_fps = np.array([
        tip_speed_schedule(V * kts).to(ft / s).magnitude
        for V in V_kts_arr])

    # ----------------------------------------------------------------
    apply_style()
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.suptitle(
        "Prouty Fig 10.4  --  Constraints on Choice of Rotor Tip Speed",
        fontsize=13, color="#c9d1d9", y=0.99)

    YLIM_MIN, YLIM_MAX = 100, 950

    # ── Forbidden zones (light hatched fills) ──────────────────────────
    # Above noise limit
    ax.fill_between(V_kts_arr, noise, YLIM_MAX,
                    alpha=0.10, color="#f78166", hatch="//",
                    edgecolor="#f78166", linewidth=0.5)
    # Below SKE limit
    ax.fill_between(V_kts_arr, YLIM_MIN, ske,
                    alpha=0.10, color="#ffa657", hatch="\\\\",
                    edgecolor="#ffa657", linewidth=0.5)
    # Compressibility: above comp line where it dips below noise limit
    comp_ceil = np.minimum(V_tip_comp, noise)
    ax.fill_between(V_kts_arr, comp_ceil, noise,
                    where=V_tip_comp < noise,
                    alpha=0.10, color="#d2a8ff", hatch="//",
                    edgecolor="#d2a8ff", linewidth=0.5)
    # Stall: below stall line where it rises above SKE limit
    stall_floor = np.maximum(V_tip_stall, ske)
    ax.fill_between(V_kts_arr, ske, stall_floor,
                    where=V_tip_stall > ske,
                    alpha=0.10, color="#3fb950", hatch="\\\\",
                    edgecolor="#3fb950", linewidth=0.5)

    # ── Constraint boundary lines ──────────────────────────────────────
    ax.axhline(noise, color="#f78166", linewidth=2.0,
               label=f"Noise Limit  ({noise:.0f} ft/s)")
    ax.axhline(ske,   color="#ffa657", linewidth=2.0,
               label=f"Stored KE Limit  ({ske:.0f} ft/s)")

    mask_comp  = V_tip_comp < noise
    if mask_comp.any():
        ax.plot(V_kts_arr[mask_comp], V_tip_comp[mask_comp],
                color="#d2a8ff", linewidth=2.0,
                label=f"Compressibility  (M = {M_comp:.2f})")

    mask_stall = (V_tip_stall > ske) & (V_tip_stall < YLIM_MAX)
    if mask_stall.any():
        ax.plot(V_kts_arr[mask_stall], V_tip_stall[mask_stall],
                color="#3fb950", linewidth=2.0,
                label=f"Stall Limit  (μ_max = {mu_max:.2f})")

    # ── Design overlays ────────────────────────────────────────────────
    ax.axhline(V_tip_hover, color="#58a6ff", linewidth=1.8,
               linestyle="--",
               label=f"Your hover tip speed  ({V_tip_hover:.0f} ft/s)")

    ax.plot(V_kts_arr, sched_fps, color="#79c0ff", linewidth=1.8,
            linestyle="-.",
            label="Engine tip-speed schedule (used in sweeps)")

    # ── Annotation labels on the chart ────────────────────────────────
    ax.text(8,  noise + 18,  "Noise Limit",
            color="#f78166", fontsize=9, va="bottom", fontweight="bold")
    ax.text(8,  ske   - 18,  "Stored Kinetic\nEnergy Limit",
            color="#ffa657", fontsize=9, va="top",    fontweight="bold")

    # Compressibility label at the midpoint of the visible portion
    if mask_comp.any():
        idx   = len(V_kts_arr[mask_comp]) // 2
        cx    = V_kts_arr[mask_comp][idx]
        cy    = V_tip_comp[mask_comp][idx]
        ax.text(cx + 5, cy + 25, "Compressibility\nLimit",
                color="#d2a8ff", fontsize=9, ha="left", fontweight="bold")

    # Stall label
    if mask_stall.any():
        idx   = int(len(V_kts_arr[mask_stall]) * 0.6)
        sx    = V_kts_arr[mask_stall][idx]
        sy    = V_tip_stall[mask_stall][idx]
        ax.text(sx + 4, sy - 30, "Stall Limit",
                color="#3fb950", fontsize=9, ha="left", fontweight="bold")

    # "Acceptable" region  --  find the centre of the valid envelope
    acc_mid_V = 80
    acc_mid_T = (noise + ske) / 2
    ax.text(acc_mid_V, acc_mid_T, "Acceptable Tip\nSpeed Choices",
            color="#c9d1d9", fontsize=12, ha="center", va="center",
            fontweight="bold", alpha=0.65)

    ax.set_xlim(0, 220)
    ax.set_ylim(YLIM_MIN, YLIM_MAX)
    ax.set_xticks(range(0, 221, 20))
    ax.set_xlabel("Forward Speed  (knots)", fontsize=12)
    ax.set_ylabel("Rotor Tip Speed  (feet per second)", fontsize=12)
    ax.set_title(
        "Shaded regions = forbidden zones for each constraint.  "
        "Advancing-tip Mach limit is M_comp \u00d7 a\u2080 \u2212 V_fwd.\n"
        "Your hover tip speed (blue dashed) and the engine tip-speed "
        "schedule (dot-dash) are overlaid for reference.",
        fontsize=9, color="#8b949e")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Output window  --  fixed horizontal scrollbar
# ---------------------------------------------------------------------------
class OutputWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Marine One eVTOL -- Analysis Output")
        self.configure(bg="#0d1117")
        self.geometry("880x800")
        self.minsize(600, 400)

        # Top bar: title + save button
        top_bar = tk.Frame(self, bg="#0d1117")
        top_bar.pack(fill="x")
        tk.Label(top_bar, text="  ANALYSIS OUTPUT",
                 bg="#0d1117", fg="#58a6ff",
                 font=("Consolas", 13, "bold"), pady=8).pack(side="left")
        ttk.Button(top_bar, text=" Save to file ",
                   style="Save.TButton",
                   command=self._save).pack(side="right", padx=10, pady=4)

        # Wrapper frame so both scrollbars are managed together
        frame = tk.Frame(self, bg="#0d1117")
        frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        vscroll = ttk.Scrollbar(frame, orient="vertical")
        xscroll = ttk.Scrollbar(frame, orient="horizontal")

        self.text = tk.Text(
            frame, wrap="none",
            bg="#161b22", fg="#c9d1d9",
            font=("Consolas", 11),
            insertbackground="#c9d1d9",
            relief="flat", borderwidth=2,
            yscrollcommand=vscroll.set,
            xscrollcommand=xscroll.set,
            state="disabled")

        vscroll.config(command=self.text.yview)
        xscroll.config(command=self.text.xview)

        # Pack order matters: scrollbars first so they get their share of space
        xscroll.pack(side="bottom", fill="x")
        vscroll.pack(side="right",  fill="y")
        self.text.pack(fill="both", expand=True)

    def set_text(self, content: str):
        self.text.configure(state="normal")
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, content)
        self.text.configure(state="disabled")
        self.text.see("1.0")
        self.lift()
        self.focus_force()

    def _save(self):
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Save Analysis Output",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile="evtol_output.txt")
        if not path:
            return
        try:
            self.text.configure(state="normal")
            content = self.text.get("1.0", tk.END)
            self.text.configure(state="disabled")
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            messagebox.showerror("Save Error", str(e), parent=self)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Marine One eVTOL -- Interactive Design Tool")
        self.configure(bg="#0d1117")
        self.resizable(True, True)
        self._last    = None
        self._out_win = None
        self._style()
        self._ui()

    # ── ttk styles ─────────────────────────────────────────────────────
    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        bg = "#0d1117"; fr = "#161b22"; fg = "#c9d1d9"
        ac = "#58a6ff"; en = "#21262d"
        s.configure("TFrame",      background=bg)
        s.configure("TLabel",      background=bg, foreground=fg,
                    font=("Consolas", 10))
        s.configure("TLabelframe", background=fr, foreground=ac,
                    font=("Consolas", 10, "bold"),
                    bordercolor="#30363d", relief="solid")
        s.configure("TLabelframe.Label", background=fr, foreground=ac,
                    font=("Consolas", 10, "bold"))
        s.configure("TEntry",      fieldbackground=en, foreground=fg,
                    insertcolor=fg, font=("Consolas", 10))
        s.configure("Run.TButton",  background=ac,       foreground="#0d1117",
                    font=("Consolas", 11, "bold"), padding=8)
        s.configure("Plot.TButton", background="#3fb950", foreground="#0d1117",
                    font=("Consolas", 11, "bold"), padding=8)
        s.configure("Out.TButton",  background="#d2a8ff", foreground="#0d1117",
                    font=("Consolas", 11, "bold"), padding=8)
        s.configure("Tip.TButton",  background="#ffa657", foreground="#0d1117",
                    font=("Consolas", 10, "bold"), padding=6)
        s.configure("Save.TButton", background="#8b949e", foreground="#0d1117",
                    font=("Consolas", 10, "bold"), padding=6)
        s.map("Run.TButton",  background=[("active", "#79c0ff")])
        s.map("Plot.TButton", background=[("active", "#56d364")])
        s.map("Out.TButton",  background=[("active", "#e8c8ff")])
        s.map("Tip.TButton",  background=[("active", "#ffcf86")])
        s.map("Save.TButton", background=[("active", "#c9d1d9")])

    # ── Top-level layout ───────────────────────────────────────────────
    def _ui(self):
        BG = "#0d1117"
        tk.Label(self,
                 text="  Marine One eVTOL -- Interactive Design Tool",
                 bg=BG, fg="#58a6ff",
                 font=("Consolas", 14, "bold"), pady=5).pack(fill="x")

        # Row 1: Design | Performance
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=(2, 0))
        self._design_sec(top)
        self._perf_sec(top)

        # Row 2: Sweep & Tip Speed (full width)
        mid = ttk.Frame(self)
        mid.pack(fill="x", padx=10, pady=(2, 0))
        self._sweep_sec(mid)

        # Action buttons
        bf = ttk.Frame(self)
        bf.pack(pady=5)
        ttk.Button(bf, text="  RUN ANALYSIS  ", style="Run.TButton",
                   command=self._run).pack(side="left", padx=10)
        ttk.Button(bf, text="  SHOW OUTPUT  ", style="Out.TButton",
                   command=self._show_out).pack(side="left", padx=10)
        ttk.Button(bf, text="  SHOW PLOTS  ", style="Plot.TButton",
                   command=self._plots).pack(side="left", padx=10)
        # Separator
        ttk.Separator(bf, orient="vertical").pack(
            side="left", fill="y", padx=8, pady=2)
        ttk.Button(bf, text=" SAVE INPUTS ", style="Save.TButton",
                   command=self._save_inputs).pack(side="left", padx=4)
        ttk.Button(bf, text=" LOAD INPUTS ", style="Save.TButton",
                   command=self._load_inputs).pack(side="left", padx=4)
        ttk.Button(bf, text=" SAVE OUTPUT ", style="Save.TButton",
                   command=self._save_output).pack(side="left", padx=4)

        # Status bar
        self.status = tk.StringVar(value="  Ready -- press RUN ANALYSIS")
        tk.Label(self, textvariable=self.status,
                 bg="#161b22", fg="#8b949e",
                 font=("Consolas", 10), anchor="w",
                 padx=10, pady=5).pack(fill="x", padx=10, pady=(0, 6))

    # ── Shared label+entry helper ──────────────────────────────────────
    def _le(self, parent, row, col, label, default, width=8):
        ttk.Label(parent, text=label).grid(
            row=row, column=col, sticky="e", padx=(4, 2), pady=2)
        v = tk.StringVar(value=str(default))
        ttk.Entry(parent, textvariable=v, width=width).grid(
            row=row, column=col + 1, sticky="w", padx=(2, 4), pady=2)
        return v

    # ── Design Characteristics panel ─────────────────────────────────
    def _design_sec(self, parent):
        f = ttk.LabelFrame(parent, text="  DESIGN CHARACTERISTICS  ", padding=(8, 4))
        f.pack(side="left", fill="both", expand=True, padx=4, pady=2)
        r = 0
        self.v_gw     = self._le(f, r, 0, "Gross Weight Guess [lb]", _D_GW,    9); r += 1
        self.v_radius = self._le(f, r, 0, "Rotor Radius [ft]",       _D_RADIUS, 7); r += 1
        self.v_blades = self._le(f, r, 0, "Number of Blades",        _D_BLADES, 5); r += 1
        self.v_wheels = self._le(f, r, 0, "Number of Wheels",        _D_WHEELS, 5); r += 1
        self.v_chord  = self._le(f, r, 0, "Blade Chord [ft]",        _D_CHORD,  7); r += 1
        ttk.Label(f, text="--- Cockpit Geometry ---").grid(
            row=r, column=0, columnspan=2, pady=(4, 1)); r += 1
        self.v_ck_l = self._le(f, r, 0, "Length [ft]", _D_CK_L, 6); r += 1
        self.v_ck_w = self._le(f, r, 0, "Width  [ft]", _D_CK_W, 6); r += 1
        self.v_ck_h = self._le(f, r, 0, "Height [ft]", _D_CK_H, 6); r += 1
        ttk.Label(f, text="--- Cabin Geometry ---").grid(
            row=r, column=0, columnspan=2, pady=(4, 1)); r += 1
        self.v_ca_l = self._le(f, r, 0, "Length [ft]", _D_CA_L, 6); r += 1
        self.v_ca_w = self._le(f, r, 0, "Width  [ft]", _D_CA_W, 6); r += 1
        self.v_ca_h = self._le(f, r, 0, "Height [ft]", _D_CA_H, 6); r += 1
        ttk.Label(f, text="--- Volume Sizing ---",
                  foreground="#ffa657").grid(
            row=r, column=0, columnspan=2, pady=(4, 1)); r += 1
        self.v_npeop  = self._le(f, r, 0, "N_people (crew+pax)",    _D_NPEOP,  5); r += 1
        self.v_k_elec = self._le(f, r, 0, "K_electronics [ft^3/lb]", _D_K_ELEC, 7); r += 1
        self.v_k_batt = self._le(f, r, 0, "K_battery [ft^3/lb]",     _D_K_BATT, 7); r += 1

    # ── Performance Characteristics panel ────────────────────────────
    def _perf_sec(self, parent):
        f = ttk.LabelFrame(parent,
                           text="  PERFORMANCE CHARACTERISTICS  ",
                           padding=(8, 4))
        f.pack(side="left", fill="both", expand=True, padx=4, pady=2)
        r = 0
        self.v_se      = self._le(f, r, 0, "Battery Spec Energy [Wh/kg]", _D_SE,      9); r += 1
        self.v_vfwd    = self._le(f, r, 0, "Cruise Speed [kts]",           _D_VFWD,    7); r += 1
        self.v_aoa     = self._le(f, r, 0, "Angle of Attack [deg]",        _D_AOA,     7); r += 1
        self.v_climb   = self._le(f, r, 0, "Climb Rate [ft/min]",          _D_CLIMB,   7); r += 1
        self.v_descent = self._le(f, r, 0, "Descent Rate [ft/min]",        _D_DESCENT, 7); r += 1
        self.v_rpm     = self._le(f, r, 0, "Hover Rotor Speed [RPM]",      _D_RPM,     7); r += 1

        ttk.Label(f, text="--- Mission Profile ---",
                  foreground="#ffa657").grid(
            row=r, column=0, columnspan=2, pady=(4, 1)); r += 1
        self.v_range      = self._le(f, r, 0, "Range [nmi]",          _D_RANGE,   7); r += 1
        self.v_cruise_alt = self._le(f, r, 0, "Cruise Alt [ft]",      _D_ALT,     7); r += 1
        self.v_hover_time = self._le(f, r, 0, "Hover Time [min/end]", _D_HOVER_T, 7); r += 1

        ttk.Label(f, text="--- Payload [lb] ---",
                  foreground="#ffa657").grid(
            row=r, column=0, columnspan=2, pady=(4, 1)); r += 1
        self.v_pay_crew = self._le(f, r, 0, "Crew & Passengers", _D_CREW, 7); r += 1
        self.v_pay_elec = self._le(f, r, 0, "Electronics",       _D_ELEC, 7); r += 1
        self.v_pay_def  = self._le(f, r, 0, "Defense System",    _D_DEF,  7); r += 1

    # ── Sweep & Tip Speed panel (full-width row 2, 2-row compact grid) ──
    def _sweep_sec(self, parent):
        f = ttk.LabelFrame(parent,
                           text="  SWEEP & TIP SPEED CONSTRAINTS  ",
                           padding=(8, 4))
        f.pack(fill="x", padx=6, pady=(0, 2))

        py = 2   # vertical padding kept tight throughout

        # ── Row 0: velocity & altitude sweep ranges ──────────────────────
        c = 0
        ttk.Label(f, text="Sweeps:", foreground="#ffa657",
                  font=("Consolas", 9, "bold")).grid(
            row=0, column=c, sticky="e", padx=(2, 4), pady=py); c += 1

        # ROC range
        ttk.Label(f, text="ROC [kts]").grid(
            row=0, column=c, sticky="e", padx=(6, 2), pady=py); c += 1
        self.v_roc_vmin = tk.StringVar(value=str(_D_ROC_VMIN))
        ttk.Entry(f, textvariable=self.v_roc_vmin, width=4).grid(
            row=0, column=c, sticky="w", padx=1, pady=py); c += 1
        ttk.Label(f, text="-").grid(row=0, column=c, padx=1); c += 1
        self.v_roc_vmax = tk.StringVar(value=str(_D_ROC_VMAX))
        ttk.Entry(f, textvariable=self.v_roc_vmax, width=4).grid(
            row=0, column=c, sticky="w", padx=1, pady=py); c += 1

        # Vmax range
        ttk.Label(f, text="Vmax [kts]").grid(
            row=0, column=c, sticky="e", padx=(10, 2), pady=py); c += 1
        self.v_vmax_vmin = tk.StringVar(value=str(_D_VMAX_VMIN))
        ttk.Entry(f, textvariable=self.v_vmax_vmin, width=4).grid(
            row=0, column=c, sticky="w", padx=1, pady=py); c += 1
        ttk.Label(f, text="-").grid(row=0, column=c, padx=1); c += 1
        self.v_vmax_vmax = tk.StringVar(value=str(_D_VMAX_VMAX))
        ttk.Entry(f, textvariable=self.v_vmax_vmax, width=4).grid(
            row=0, column=c, sticky="w", padx=1, pady=py); c += 1

        # Altitude sweep
        ttk.Label(f, text="Alt max [ft]").grid(
            row=0, column=c, sticky="e", padx=(10, 2), pady=py); c += 1
        self.v_alt_max = tk.StringVar(value=str(_D_ALT_MAX))
        ttk.Entry(f, textvariable=self.v_alt_max, width=6).grid(
            row=0, column=c, sticky="w", padx=1, pady=py); c += 1
        ttk.Label(f, text="pts").grid(row=0, column=c, padx=(4, 1)); c += 1
        self.v_alt_pts = tk.StringVar(value=str(_D_ALT_PTS))
        ttk.Entry(f, textvariable=self.v_alt_pts, width=3).grid(
            row=0, column=c, sticky="w", padx=1, pady=py); c += 1

        # Fig 10.4 hint (top-right, spanning button column)
        ttk.Label(f, text="tip speed vs. fwd speed",
                  foreground="#8b949e",
                  font=("Consolas", 8)).grid(
            row=0, column=c, columnspan=2,
            sticky="w", padx=(16, 4), pady=py)

        # ── Row 1: tip speed constraint inputs + chart button ────────────
        c = 0
        ttk.Label(f, text="Tip limits:", foreground="#ffa657",
                  font=("Consolas", 9, "bold")).grid(
            row=1, column=c, sticky="e", padx=(2, 4), pady=py); c += 1

        self.v_tip_noise = tk.StringVar(value=str(_D_TIP_NOISE))
        self.v_tip_ske   = tk.StringVar(value=str(_D_TIP_SKE))
        self.v_tip_M     = tk.StringVar(value=str(_D_TIP_M_COMP))
        self.v_tip_mu    = tk.StringVar(value=str(_D_TIP_MU_MAX))

        for lbl, var in [
            ("Noise [ft/s]",  self.v_tip_noise),
            ("SKE [ft/s]",    self.v_tip_ske),
            ("Comp. Mach",    self.v_tip_M),
            (u"Max \u03bc",   self.v_tip_mu),
        ]:
            ttk.Label(f, text=lbl).grid(
                row=1, column=c, sticky="e", padx=(6, 2), pady=py); c += 1
            ttk.Entry(f, textvariable=var, width=5).grid(
                row=1, column=c, sticky="w", padx=1, pady=py)
            c += 2   # skip the "-" column used in sweep rows above

        # Chart button (aligned under the hint text)
        ttk.Button(f, text="  View Fig 10.4  ",
                   style="Tip.TButton",
                   command=self._show_tip_chart).grid(
            row=1, column=c, columnspan=2,
            sticky="w", padx=(14, 4), pady=py)

    # ── Collect all inputs ────────────────────────────────────────────
    def _inputs(self) -> dict:
        return {
            # Design
            "gw_guess":    _fval(self.v_gw,     _D_GW),
            "rotor_radius":_fval(self.v_radius,  _D_RADIUS),
            "n_blades":    _ival(self.v_blades,  _D_BLADES),
            "n_wheels":    _ival(self.v_wheels,  _D_WHEELS),
            "chord":       _fval(self.v_chord,   _D_CHORD),
            "ck_l":        _fval(self.v_ck_l,    _D_CK_L),
            "ck_w":        _fval(self.v_ck_w,    _D_CK_W),
            "ck_h":        _fval(self.v_ck_h,    _D_CK_H),
            "ca_l":        _fval(self.v_ca_l,    _D_CA_L),
            "ca_w":        _fval(self.v_ca_w,    _D_CA_W),
            "ca_h":        _fval(self.v_ca_h,    _D_CA_H),
            "n_people":    _ival(self.v_npeop,   _D_NPEOP),
            "k_elec":      _fval(self.v_k_elec,  _D_K_ELEC),
            "k_batt":      _fval(self.v_k_batt,  _D_K_BATT),
            # Performance
            "battery_se":  _fval(self.v_se,      _D_SE),
            "cruise_speed":_fval(self.v_vfwd,    _D_VFWD),
            "aoa_deg":     _fval(self.v_aoa,     _D_AOA),
            "climb_rate":  _fval(self.v_climb,   _D_CLIMB),
            "descent_rate":_fval(self.v_descent, _D_DESCENT),
            "rotor_rpm":   _fval(self.v_rpm,     _D_RPM),
            # Mission
            "range_nmi":      _fval(self.v_range,      _D_RANGE),
            "cruise_alt_ft":  _fval(self.v_cruise_alt, _D_ALT),
            "hover_time_min": _fval(self.v_hover_time, _D_HOVER_T),
            # Payload
            "payload_crew":_fval(self.v_pay_crew, _D_CREW),
            "payload_elec":_fval(self.v_pay_elec, _D_ELEC),
            "payload_def": _fval(self.v_pay_def,  _D_DEF),
            # Sweep
            "roc_vmin":  _fval(self.v_roc_vmin,  _D_ROC_VMIN),
            "roc_vmax":  _fval(self.v_roc_vmax,  _D_ROC_VMAX),
            "vmax_vmin": _fval(self.v_vmax_vmin, _D_VMAX_VMIN),
            "vmax_vmax": _fval(self.v_vmax_vmax, _D_VMAX_VMAX),
            "alt_max":   _fval(self.v_alt_max,   _D_ALT_MAX),
            "alt_pts":   _ival(self.v_alt_pts,   _D_ALT_PTS),
            # Tip speed constraints (for Fig 10.4 display)
            "tip_noise":  _fval(self.v_tip_noise, _D_TIP_NOISE),
            "tip_ske":    _fval(self.v_tip_ske,   _D_TIP_SKE),
            "tip_M_comp": _fval(self.v_tip_M,     _D_TIP_M_COMP),
            "tip_mu_max": _fval(self.v_tip_mu,    _D_TIP_MU_MAX),
        }

    # ── Run analysis ──────────────────────────────────────────────────
    def _run(self):
        self.status.set("  Running analysis ...  (may take a few seconds)")
        self.update_idletasks()
        try:
            res = run_analysis(self._inputs())
            self._last = res
            txt = format_output(res)
            self._open_out(txt)
            dw = (res["gw"].magnitude
                  - res["ref"]["new_mtow"].to(lbf).magnitude)
            self.status.set(
                f"  Done -- DW = {dw:+,.0f} lb  |  SHOW OUTPUT / SHOW PLOTS")
        except Exception as e:
            self.status.set(f"  Error: {e}")
            messagebox.showerror("Analysis Error", str(e))
            raise

    # ── Output window helpers ─────────────────────────────────────────
    def _open_out(self, txt: str):
        if self._out_win is None or not self._out_win.winfo_exists():
            self._out_win = OutputWindow(self)
        self._out_win.set_text(txt)

    def _show_out(self):
        if not self._last:
            messagebox.showinfo("No Data", "Run analysis first.")
            return
        self._open_out(format_output(self._last))

    # ── Plots ─────────────────────────────────────────────────────────
    def _plots(self):
        if not self._last:
            messagebox.showinfo("No Data", "Run analysis first.")
            return
        r   = self._last
        ref = r["ref"]
        # Supply a 2-point convergence history when single-pass was used
        if "convergence_history" not in ref:
            ref["convergence_history"] = np.array(
                [r["gw"].magnitude, ref["new_mtow"].to(lbf).magnitude])
        if "mtow_converged" not in ref:
            ref["mtow_converged"] = ref["new_mtow"]
        generate_all_plots(
            result=ref, design=r["design"], cg_seq=r["cg_seq"],
            altitudes_ft=r["altitudes"], roc=r["roc"], v_max=r["v_max"])

    # ── Save / Load inputs ──────────────────────────────────────────
    def _var_map(self) -> dict:
        """Return {input_key: StringVar} for every editable field."""
        return {
            "gw_guess":    self.v_gw,        "rotor_radius": self.v_radius,
            "n_blades":    self.v_blades,     "n_wheels":     self.v_wheels,
            "chord":       self.v_chord,
            "ck_l": self.v_ck_l, "ck_w": self.v_ck_w, "ck_h": self.v_ck_h,
            "ca_l": self.v_ca_l, "ca_w": self.v_ca_w, "ca_h": self.v_ca_h,
            "n_people":    self.v_npeop,
            "k_elec":      self.v_k_elec,    "k_batt":       self.v_k_batt,
            "battery_se":  self.v_se,        "cruise_speed":  self.v_vfwd,
            "aoa_deg":     self.v_aoa,       "climb_rate":    self.v_climb,
            "descent_rate":self.v_descent,   "rotor_rpm":     self.v_rpm,
            "range_nmi":   self.v_range,     "cruise_alt_ft": self.v_cruise_alt,
            "hover_time_min": self.v_hover_time,
            "payload_crew":self.v_pay_crew,  "payload_elec":  self.v_pay_elec,
            "payload_def": self.v_pay_def,
            "roc_vmin":    self.v_roc_vmin,  "roc_vmax":      self.v_roc_vmax,
            "vmax_vmin":   self.v_vmax_vmin, "vmax_vmax":     self.v_vmax_vmax,
            "alt_max":     self.v_alt_max,   "alt_pts":       self.v_alt_pts,
            "tip_noise":   self.v_tip_noise, "tip_ske":       self.v_tip_ske,
            "tip_M_comp":  self.v_tip_M,     "tip_mu_max":    self.v_tip_mu,
        }

    def _save_inputs(self):
        path = filedialog.asksaveasfilename(
            title="Save Input Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="evtol_inputs.json")
        if not path:
            return
        try:
            data = self._inputs()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self.status.set(f"  Inputs saved to {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def _load_inputs(self):
        path = filedialog.askopenfilename(
            title="Load Input Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            vm = self._var_map()
            loaded = 0
            for key, var in vm.items():
                if key in data:
                    var.set(str(data[key]))
                    loaded += 1
            self.status.set(
                f"  Loaded {loaded} inputs from {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _save_output(self):
        if not self._last:
            messagebox.showinfo("No Data", "Run analysis first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Analysis Output",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile="evtol_output.txt")
        if not path:
            return
        try:
            txt = format_output(self._last)
            with open(path, "w", encoding="utf-8") as f:
                f.write(txt)
            self.status.set(f"  Output saved to {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    # ── Tip speed chart ───────────────────────────────────────────────
    def _show_tip_chart(self):
        try:
            show_tip_speed_chart(self._inputs(), self._last)
        except Exception as e:
            messagebox.showerror("Chart Error", str(e))
            raise


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = App()
    app.mainloop()
