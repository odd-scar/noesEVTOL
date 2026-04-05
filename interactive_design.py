"""
interactive_design.py  —  Marine One eVTOL Interactive Design Tool
=================================================================
Uses project_2_revised.py as the computation engine.

INPUT SECTIONS
--------------
  Design Characteristics:
    GW guess, rotor geometry, fuselage LWH, N_people,
    K_electronics, K_battery  (volume-density coefficients)
  Performance Characteristics:
    battery SE, cruise speed, AoA, climb rate

OUTPUT (separate resizable window)
-----------------------------------
  1. Mission energy breakdown
  2. Battery weight
  3. Weight statement + feasibility (DW)
  4. Min battery SE required (with math)
  5. Fuselage volume sizing (Vmargin formula)
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
)

# Fixed mission parameters
CRUISE_RANGE_NMI   = 250
CRUISE_ALT_FT      = 3_000
HOVER_TIME_MIN     = 2.0
PAYLOAD_CREW_LBF   = 2_000
PAYLOAD_ELEC_LBF   = 1_000
PAYLOAD_DEF_LBF    = 1_000


def _fval(v, d):
    try: return float(v.get())
    except: return d

def _ival(v, d):
    try: return int(v.get())
    except: return d


def run_analysis(inp):
    design = RotorcraftDesign(
        rotor=RotorGeometry(
            radius=inp["rotor_radius"]*ft, chord=inp["chord"]*ft,
            n_blades=inp["n_blades"], n_wheels=inp["n_wheels"]),
        fuselage=FuselageGeometry(
            cockpit_length=inp["ck_l"]*ft, cockpit_width=inp["ck_w"]*ft,
            cockpit_height=inp["ck_h"]*ft, cabin_length=inp["ca_l"]*ft,
            cabin_width=inp["ca_w"]*ft, cabin_height=inp["ca_h"]*ft,
            landing_gear_height=3.0*ft),
        payload=PayloadWeights(
            crew_and_passengers=PAYLOAD_CREW_LBF*lbf,
            electronics=PAYLOAD_ELEC_LBF*lbf,
            defense_system=PAYLOAD_DEF_LBF*lbf),
        aero=AeroCoefficients(cd0_hover=0.001, induced_power_factor=1.15,
                              profile_power_factor=4.7, power_margin=1.25),
        battery=BatteryModel(
            specific_energy=(inp["battery_se"]/1000.0)*kWh/kg,
            usable_fraction=0.90),
        gw_guess=inp["gw_guess"]*lbf,
        hover_rotor_speed_rpm=inp["rotor_rpm"],
    )
    mission = MissionProfile(
        cruise_range=CRUISE_RANGE_NMI*nmi, cruise_altitude=CRUISE_ALT_FT*ft,
        climb_rate=inp["climb_rate"]*ft/minute,
        cruise_speed=inp["cruise_speed"]*kts,
        descent_rate=inp["descent_rate"]*ft/minute,
        hover_time_takeoff=HOVER_TIME_MIN*minute,
        hover_time_landing=HOVER_TIME_MIN*minute)

    gw  = design.gw_guess.to(lbf)
    ref = single_point_analysis(design, mission, gw, alpha_deg=inp["aoa_deg"])

    altitudes = np.linspace(0, 14_000, 50)
    p_inst_hp = ref["p_installed"].magnitude
    roc, v_best = max_rate_of_climb_vs_altitude(design, gw, p_inst_hp, altitudes)
    v_max = max_speed_vs_altitude(design, gw, p_inst_hp, altitudes)

    vol = fuselage_volume_check(
        design, ref["battery"]["weight"],
        n_occupants=inp["n_people"],
        k_electronics_ft3_lb=inp["k_elec"],
        k_battery_ft3_lb=inp["k_batt"])

    cg_seq = cg_loading_sequence(
        design, ref["component_weights"], ref["battery"]["weight"])

    return {"design": design, "mission": mission, "ref": ref, "gw": gw,
            "altitudes": altitudes, "roc": roc, "v_max": v_max,
            "cg_seq": cg_seq, "vol": vol,
            "k_elec": inp["k_elec"], "k_batt": inp["k_batt"],
            "n_people": inp["n_people"]}


def format_output(r):
    ref    = r["ref"];  design = r["design"];  gw = r["gw"]
    en     = ref["mission_energy"];  batt = ref["battery"];  vol = r["vol"]
    gw_lb      = gw.magnitude
    w_empty_lb = ref["w_empty"].to(lbf).magnitude
    w_batt_lb  = batt["weight"].to(lbf).magnitude
    w_pay_lb   = design.payload.total_payload.to(lbf).magnitude
    mtow_lb    = ref["new_mtow"].to(lbf).magnitude
    se_wh_kg   = design.battery.specific_energy.to(kWh/kg).magnitude * 1000.0
    e_nom_kWh  = en["E_nominal"].to(kWh).magnitude
    k_elec = r["k_elec"];  k_batt = r["k_batt"];  n_ppl = r["n_people"]
    dw = gw_lb - (w_empty_lb + w_pay_lb + w_batt_lb)
    w_batt_budget_lb = gw_lb - w_empty_lb - w_pay_lb
    m_batt_budget_sl = w_batt_budget_lb / 32.174
    m_batt_budget_kg = m_batt_budget_sl * 14.5939
    se_min_wh_kg = (e_nom_kWh*1000.0)/m_batt_budget_kg if m_batt_budget_kg>0 else float("inf")
    ratio = se_min_wh_kg/se_wh_kg if se_wh_kg>0 else float("inf")

    if abs(dw)<200:   fs, fy = "CLOSES  (DW ~ 0)", "[OK]"
    elif dw>0:        fs, fy = "GW GUESS TOO HIGH -- may reduce", "[^]"
    else:             fs, fy = "DOES NOT CLOSE -- GW guess too low", "[X]"

    S = "="*66;  s2 = "-"*66;  L = [];  a = L.append

    # 1. Mission energy
    a(f"\n{S}"); a("  SECTION 1 -- MISSION ENERGY BREAKDOWN"); a(s2)
    a(f"  {'Phase':<22} {'Time (min)':>10}  {'Energy (kWh)':>12}"); a(s2)
    for nm,t,e in [("Hover Takeoff",en["t_hover_takeoff"],en["E_hover_takeoff"]),
                    ("Climb",en["t_climb"],en["E_climb"]),
                    ("Cruise",en["t_cruise"],en["E_cruise"]),
                    ("Descent",en["t_descent"],en["E_descent"]),
                    ("Hover Landing",en["t_hover_landing"],en["E_hover_landing"])]:
        a(f"  {nm:<22} {t.to(minute).magnitude:>10.1f}  {e.to(kWh).magnitude:>12.1f}")
    a(s2)
    a(f"  {'Total Required':<22} {'':>10}  {en['E_required'].to(kWh).magnitude:>12.1f}")
    a(f"  {'Nominal (/ 0.90)':<22} {'':>10}  {e_nom_kWh:>12.1f}")

    # 2. Battery
    a(f"\n{S}"); a("  SECTION 2 -- BATTERY"); a(s2)
    a(f"  Specific energy (input)  : {se_wh_kg:,.0f} Wh/kg")
    a(f"  Usable fraction          : 90 %")
    a(f"  Nominal energy required  : {e_nom_kWh:,.1f} kWh")
    a(f"  Battery weight           : {w_batt_lb:,.0f} lb")

    # 3. Weight statement
    a(f"\n{S}"); a("  SECTION 3 -- WEIGHT STATEMENT"); a(s2)
    a(f"  {'Item':<42} {'Weight (lb)':>10}"); a(s2)
    a(f"  {'Assumed Gross Weight (GW_guess) [INPUT]':<42} {gw_lb:>10,.0f}"); a(s2)
    a(f"  {'Empty Weight            (computed)':<42} {w_empty_lb:>10,.0f}")
    a(f"  {'Payload Weight          (fixed)':<42} {w_pay_lb:>10,.0f}")
    a(f"    {'  Crew & Passengers':<40} {PAYLOAD_CREW_LBF:>10,.0f}")
    a(f"    {'  Electronics Bay':<40} {PAYLOAD_ELEC_LBF:>10,.0f}")
    a(f"    {'  Defensive Armament':<40} {PAYLOAD_DEF_LBF:>10,.0f}")
    a(f"  {'Battery Weight          (computed)':<42} {w_batt_lb:>10,.0f}"); a(s2)
    a(f"  {'Computed MTOW = W_empty+W_pay+W_batt':<42} {mtow_lb:>10,.0f}")

    # 4. Feasibility
    a(f"\n{S}"); a("  SECTION 4 -- FEASIBILITY CHECK"); a(s2)
    a(f"  DW = GW_guess - (W_empty + W_payload + W_battery)")
    a(f"     = {gw_lb:,.0f} - ({w_empty_lb:,.0f} + {w_pay_lb:,.0f} + {w_batt_lb:,.0f})")
    a(f"     = {gw_lb:,.0f} - {w_empty_lb+w_pay_lb+w_batt_lb:,.0f}")
    a(f"     = {dw:+,.0f} lb"); a("")
    a(f"  {fy}  {fs}")
    if dw < -200:
        a(""); a(f"  The computed MTOW ({mtow_lb:,.0f} lb) exceeds the GW guess")
        a(f"  ({gw_lb:,.0f} lb). Increase GW, reduce range, or raise battery SE.")
    elif dw > 200:
        a(""); a(f"  GW_guess is conservative. Computed MTOW = {mtow_lb:,.0f} lb.")

    # 5. Minimum SE
    a(f"\n{S}"); a("  SECTION 5 -- MINIMUM BATTERY SPECIFIC ENERGY"); a(s2)
    a(f"  Step 1: W_budget = GW - W_empty - W_payload = {w_batt_budget_lb:,.0f} lb")
    a(f"  Step 2: m_budget = {w_batt_budget_lb:,.0f}/32.174 = {m_batt_budget_sl:,.1f} sl = {m_batt_budget_kg:,.1f} kg")
    a(f"  Step 3: E_nominal = {e_nom_kWh:,.1f} kWh")
    if m_batt_budget_kg > 0:
        a(f"  Step 4: SE_min = {e_nom_kWh:,.1f} kWh / {m_batt_budget_kg:,.1f} kg = {se_min_wh_kg:,.0f} Wh/kg")
        a(f"  Current SE = {se_wh_kg:,.0f} Wh/kg  |  Ratio = {ratio:.2f}x")
        if se_min_wh_kg <= se_wh_kg: a("  [OK] Current SE satisfies the mission.")
        else: a(f"  [X] Need {ratio:.2f}x more energy density.")
    else: a("  [!] Battery budget <= 0")

    # 6. Volume sizing (per Geometry_function.pdf reference)
    a(f"\n{S}"); a("  SECTION 6 -- FUSELAGE VOLUME SIZING CHECK"); a(s2)
    a("  Formulas (from reference):")
    a("    V_usable  = Cockpit_Volume + Cabin_Volume")
    a("    V_payload = V_seats_tot + V_electronics")
    a(f"    V_seats   = N_people * (L_seat * W_seat * H_seat)")
    a(f"    V_elec    = K_elec * W_electronics")
    a(f"    V_batt    = K_batt * W_battery")
    a("    V_margin  = V_usable - (V_batt + V_payload)"); a(s2)
    a(f"  User inputs:")
    a(f"    N_people      = {n_ppl}")
    a(f"    K_electronics = {k_elec:.4f} ft^3/lb")
    a(f"    K_battery     = {k_batt:.4f} ft^3/lb"); a(s2)
    v = r["vol"]
    a(f"  {'Cockpit Volume (Lc*Wc*Hc)':<42} {v['V_cockpit_ft3']:>10.1f} ft^3")
    a(f"  {'Cabin Volume (Lcab*Wcab*Hcab)':<42} {v['V_cabin_ft3']:>10.1f} ft^3")
    a(f"  {'V_usable (total interior)':<42} {v['V_usable_ft3']:>10.1f} ft^3"); a(s2)
    a(f"  {'V_seats = {0}*(2.5*1.5*3.0)'.format(n_ppl):<42} {v['V_seats_ft3']:>10.1f} ft^3")
    a(f"  {'V_electronics = K*W_elec':<42} {v['V_electronics_ft3']:>10.1f} ft^3")
    a(f"  {'V_payload (seats+electronics)':<42} {v['V_payload_ft3']:>10.1f} ft^3")
    a(f"  {'V_battery = K_batt*W_batt':<42} {v['V_battery_ft3']:>10.1f} ft^3"); a(s2)
    a(f"  {'V_margin = V_usable-(V_batt+V_pay)':<42} {v['V_margin_ft3']:>10.1f} ft^3"); a("")
    if v["fits"]: a("  [OK] V_margin >= 0 -- components FIT the fuselage.")
    else:
        a(f"  [X] V_margin < 0 -- fuselage too small by {abs(v['V_margin_ft3']):.1f} ft^3.")
        a("       Increase cabin/cockpit dims or adjust K coefficients.")
    a(f"\n{S}")
    return "\n".join(L)


# ---------------------------------------------------------------------------
class OutputWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Marine One eVTOL -- Analysis Output")
        self.configure(bg="#0d1117"); self.geometry("860x780"); self.minsize(600,400)
        tk.Label(self, text="  ANALYSIS OUTPUT", bg="#0d1117", fg="#58a6ff",
                 font=("Consolas",13,"bold"), pady=8).pack(fill="x")
        self.text = scrolledtext.ScrolledText(
            self, wrap="none", bg="#161b22", fg="#c9d1d9",
            font=("Consolas",11), insertbackground="#c9d1d9",
            relief="flat", borderwidth=2)
        self.text.pack(fill="both", expand=True, padx=10, pady=(0,10))
        xscroll = ttk.Scrollbar(self.text, orient="horizontal", command=self.text.xview)
        self.text.configure(xscrollcommand=xscroll.set)
        xscroll.pack(side="bottom", fill="x")

    def set_text(self, content):
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, content)
        self.text.see("1.0"); self.lift(); self.focus_force()


# ---------------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Marine One eVTOL -- Interactive Design Tool")
        self.configure(bg="#0d1117"); self.resizable(True, True)
        self._last = None; self._out_win = None
        self._style(); self._ui()

    def _style(self):
        s = ttk.Style(self); s.theme_use("clam")
        bg="#0d1117"; fr="#161b22"; fg="#c9d1d9"; ac="#58a6ff"; en="#21262d"
        s.configure("TFrame",      background=bg)
        s.configure("TLabel",      background=bg, foreground=fg, font=("Consolas",10))
        s.configure("TLabelframe", background=fr, foreground=ac,
                    font=("Consolas",10,"bold"), bordercolor="#30363d", relief="solid")
        s.configure("TLabelframe.Label", background=fr, foreground=ac,
                    font=("Consolas",10,"bold"))
        s.configure("TEntry", fieldbackground=en, foreground=fg,
                    insertcolor=fg, font=("Consolas",10))
        s.configure("Run.TButton",  background=ac,      foreground="#0d1117",
                    font=("Consolas",11,"bold"), padding=8)
        s.configure("Plot.TButton", background="#3fb950", foreground="#0d1117",
                    font=("Consolas",11,"bold"), padding=8)
        s.configure("Out.TButton",  background="#d2a8ff", foreground="#0d1117",
                    font=("Consolas",11,"bold"), padding=8)
        s.map("Run.TButton",  background=[("active","#79c0ff")])
        s.map("Plot.TButton", background=[("active","#56d364")])
        s.map("Out.TButton",  background=[("active","#e8c8ff")])

    def _ui(self):
        BG="#0d1117"
        tk.Label(self, text="  Marine One eVTOL -- Interactive Design Tool",
                 bg=BG, fg="#58a6ff", font=("Consolas",14,"bold"), pady=10).pack(fill="x")
        top = ttk.Frame(self); top.pack(fill="x", padx=12, pady=4)
        self._design_sec(top); self._perf_sec(top)
        bf = ttk.Frame(self); bf.pack(pady=8)
        ttk.Button(bf, text="  RUN ANALYSIS  ", style="Run.TButton",
                   command=self._run).pack(side="left", padx=10)
        ttk.Button(bf, text="  SHOW OUTPUT  ", style="Out.TButton",
                   command=self._show_out).pack(side="left", padx=10)
        ttk.Button(bf, text="  SHOW PLOTS  ", style="Plot.TButton",
                   command=self._plots).pack(side="left", padx=10)
        self.status = tk.StringVar(value="  Ready -- press RUN ANALYSIS")
        tk.Label(self, textvariable=self.status, bg="#161b22", fg="#8b949e",
                 font=("Consolas",10), anchor="w", padx=10, pady=8).pack(fill="x", padx=12, pady=(0,12))

    def _le(self, p, r, c, lbl, dflt, w=8):
        ttk.Label(p, text=lbl).grid(row=r, column=c, sticky="e", padx=4, pady=3)
        v = tk.StringVar(value=str(dflt))
        ttk.Entry(p, textvariable=v, width=w).grid(row=r, column=c+1, sticky="w", padx=4, pady=3)
        return v

    def _design_sec(self, parent):
        f = ttk.LabelFrame(parent, text="  DESIGN CHARACTERISTICS  ", padding=10)
        f.pack(side="left", fill="both", expand=True, padx=6, pady=4)
        r=0
        self.v_gw     = self._le(f,r,0,"Gross Weight Guess [lb]",25000,9);r+=1
        self.v_radius = self._le(f,r,0,"Rotor Radius [ft]",19.5,7);r+=1
        self.v_blades = self._le(f,r,0,"Number of Blades",4,5);r+=1
        self.v_wheels = self._le(f,r,0,"Number of Wheels",3,5);r+=1
        self.v_chord  = self._le(f,r,0,"Blade Chord [ft]",2.0,7);r+=1
        ttk.Label(f, text="--- Cockpit Geometry ---").grid(row=r, column=0, columnspan=2, pady=(8,2));r+=1
        self.v_ck_l = self._le(f,r,0,"Length [ft]",10,6);r+=1
        self.v_ck_w = self._le(f,r,0,"Width  [ft]",8,6);r+=1
        self.v_ck_h = self._le(f,r,0,"Height [ft]",6,6);r+=1
        ttk.Label(f, text="--- Cabin Geometry ---").grid(row=r, column=0, columnspan=2, pady=(8,2));r+=1
        self.v_ca_l = self._le(f,r,0,"Length [ft]",16,6);r+=1
        self.v_ca_w = self._le(f,r,0,"Width  [ft]",8,6);r+=1
        self.v_ca_h = self._le(f,r,0,"Height [ft]",6,6);r+=1
        ttk.Label(f, text="--- Volume Sizing Inputs ---", foreground="#ffa657").grid(
            row=r, column=0, columnspan=2, pady=(8,2));r+=1
        self.v_npeop  = self._le(f,r,0,"N_people (crew+pax)",10,5);r+=1
        self.v_k_elec = self._le(f,r,0,"K_electronics [ft^3/lb]",0.020,7);r+=1
        self.v_k_batt = self._le(f,r,0,"K_battery [ft^3/lb]",0.025,7);r+=1

    def _perf_sec(self, parent):
        f = ttk.LabelFrame(parent, text="  PERFORMANCE CHARACTERISTICS  ", padding=10)
        f.pack(side="left", fill="both", expand=True, padx=6, pady=4)
        r=0
        self.v_se      = self._le(f,r,0,"Battery Spec Energy [Wh/kg]",300,9);r+=1
        self.v_vfwd    = self._le(f,r,0,"Cruise Speed [kts]",150,7);r+=1
        self.v_aoa     = self._le(f,r,0,"Angle of Attack [deg]",10,7);r+=1
        self.v_climb   = self._le(f,r,0,"Climb Rate [ft/min]",500,7);r+=1
        self.v_descent = self._le(f,r,0,"Descent Rate [ft/min]",500,7);r+=1
        self.v_rpm     = self._le(f,r,0,"Hover Rotor Speed [RPM]",300,7);r+=1
        ttk.Label(f, text="--- Fixed Mission ---").grid(row=r, column=0, columnspan=2, pady=(12,2));r+=1
        for t in [f"Range        : {CRUISE_RANGE_NMI} nmi",
                  f"Cruise alt   : {CRUISE_ALT_FT:,} ft",
                  f"Hover time   : {HOVER_TIME_MIN:.0f} min each end",
                  f"Payload      : {PAYLOAD_CREW_LBF:,} lb crew/pax",
                  f"             + {PAYLOAD_ELEC_LBF:,} lb electronics",
                  f"             + {PAYLOAD_DEF_LBF:,} lb defense"]:
            ttk.Label(f, text=t, foreground="#8b949e").grid(row=r, column=0, columnspan=2, sticky="w", pady=1);r+=1

    def _inputs(self):
        return {"gw_guess":_fval(self.v_gw,25000), "rotor_radius":_fval(self.v_radius,19.5),
                "n_blades":_ival(self.v_blades,4), "n_wheels":_ival(self.v_wheels,3),
                "chord":_fval(self.v_chord,2.0),
                "ck_l":_fval(self.v_ck_l,10), "ck_w":_fval(self.v_ck_w,8), "ck_h":_fval(self.v_ck_h,6),
                "ca_l":_fval(self.v_ca_l,16), "ca_w":_fval(self.v_ca_w,8), "ca_h":_fval(self.v_ca_h,6),
                "battery_se":_fval(self.v_se,300), "cruise_speed":_fval(self.v_vfwd,150),
                "aoa_deg":_fval(self.v_aoa,10), "climb_rate":_fval(self.v_climb,500),
                "descent_rate":_fval(self.v_descent,500), "rotor_rpm":_fval(self.v_rpm,300),
                "n_people":_ival(self.v_npeop,10),
                "k_elec":_fval(self.v_k_elec,0.020), "k_batt":_fval(self.v_k_batt,0.025)}

    def _run(self):
        self.status.set("  Running analysis ..."); self.update_idletasks()
        try:
            res = run_analysis(self._inputs()); self._last = res
            txt = format_output(res); self._open_out(txt)
            dw = res['gw'].magnitude - res['ref']['new_mtow'].to(lbf).magnitude
            self.status.set(f"  Done -- DW = {dw:+,.0f} lb  |  SHOW OUTPUT / SHOW PLOTS")
        except Exception as e:
            self.status.set(f"  Error: {e}"); messagebox.showerror("Error",str(e)); raise

    def _open_out(self, txt):
        if self._out_win is None or not self._out_win.winfo_exists():
            self._out_win = OutputWindow(self)
        self._out_win.set_text(txt)

    def _show_out(self):
        if not self._last: messagebox.showinfo("No Data","Run analysis first."); return
        self._open_out(format_output(self._last))

    def _plots(self):
        if not self._last: messagebox.showinfo("No Data","Run analysis first."); return
        r = self._last; ref = r["ref"]
        ref["convergence_history"] = np.array([r["gw"].magnitude, ref["new_mtow"].to(lbf).magnitude])
        ref["mtow_converged"] = ref["new_mtow"]
        generate_all_plots(result=ref, design=r["design"], cg_seq=r["cg_seq"],
                           altitudes_ft=r["altitudes"], roc=r["roc"], v_max=r["v_max"])


if __name__ == "__main__":
    app = App(); app.mainloop()
