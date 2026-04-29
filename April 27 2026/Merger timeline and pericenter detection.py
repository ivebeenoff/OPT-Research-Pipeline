# “””

# SECTION 24 — MERGER TIMELINE & PERICENTRE DETECTION

Author  : Abhinav Vatsa

Continuation of the MW–M31 analysis pipeline.  All globals (SNAPSHOTS, ns,
R_BINS, nb_sph, r_mid_sph, OUT_DIR, MASS_UNIT_MSUN, MIN_PART_SHELL,
G_KPC_KMS2_MSUN, PROFILE_INDICES, PROFILE_LABELS, PROFILE_COLORS, time_arr,
time_label, time_is_gyr, tmpdir, PTYPE, load_snapshot_particles, CenterOfMass)
are inherited and must be defined before this section is executed.

## Physical motivation

The merger timeline is the backbone that contextualises every other diagnostic
in this pipeline.  Without knowing *when* the first pericentre occurs, when
the galaxies coalesce, and how long the remnant takes to virialise, the
heatmaps and profiles from Sections 21–23 have no temporal anchor.

This section extracts the full orbital history of the MW–M31 pair by tracking
their centre-of-mass separation d(t) and relative velocity v_rel(t) across
all 801 snapshots, then automatically detects:

• Pericentre passages    — local minima in d(t)
• Apocentre passages     — local maxima in d(t)
• Coalescence epoch      — the snapshot where the two COMs merge
(d < d_coal threshold permanently)
• Virialisation epoch    — where the Jeans disequilibrium score (from §15)
falls below a threshold for the last time

Derived quantities per identified event:
• Pericentre distance r_peri  [kpc]
• Pericentre speed v_peri     [km/s]
• Apocentre distance r_apo    [kpc]
• Orbital period T            [Gyr]
• Specific orbital energy E_orb = ½v²_rel − G(M_MW+M_M31)/d
• Specific angular momentum   L_orb = d × v_perp

## Outputs

section24_separation.png         d(t) with pericentre/apocentre annotations
section24_phase_diagram.png      (d, v_rel) orbital phase diagram
section24_orbital_energy.png     E_orb(t) and L_orb(t)
section24_events_table.png       Formatted event table figure
section24_timeline_banner.png    Horizontal timeline banner figure
section24_summary_panel.png      Master 3-panel summary

===============================================================================
“””

import numpy as np
import matplotlib
matplotlib.use(“Agg”)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.signal import find_peaks, savgol_filter
import os
import time
import warnings

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §24.0 — CONFIGURATION                                                     ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# Distance threshold below which the two galaxy COMs are considered merged.

D_COAL_KPC        = 10.0    # [kpc]

# Savitzky-Golay smoothing window for d(t) before peak detection.

# Must be odd.  Larger = smoother but may miss sharp pericentre passages.

SGF_WINDOW        = 21
SGF_POLY          = 3

# Minimum prominence for a pericentre to be considered real (not noise).

# In units of kpc — pericentre dips shallower than this are ignored.

PERI_PROMINENCE   = 20.0    # [kpc]
APO_PROMINENCE    = 20.0    # [kpc]

# Minimum separation of two consecutive pericentres in snapshots.

PERI_MIN_SEP_SNAPS = 20

# Jeans disequilibrium score threshold for virialisation detection.

# Requires jeans_score array from §15 of the kinematics pipeline.

JEANS_VIRIAL_THRESH = 0.3

# Approximate Myr per snapshot — used for physical time conversions.

# Adjust to the actual cadence of the simulation.

MYR_PER_SNAP      = 10.0

print(”\n” + “=”*80)
print(”  SECTION 24 · Merger Timeline & Pericentre Detection”)
print(”=”*80)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §24.1 — COMPUTE SEPARATION AND RELATIVE VELOCITY                         ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

print(”\n  Computing COM trajectories across all snapshots …”)

t0 = time.perf_counter()

com_mw_pos  = np.full((ns, 3), np.nan)   # MW COM position in joint COM frame
com_m31_pos = np.full((ns, 3), np.nan)   # M31 COM position
separation  = np.full(ns, np.nan)        # |r_M31 − r_MW|  [kpc]

for i, snap_num in enumerate(SNAPSHOTS):

```
mw_file  = os.path.join(tmpdir, f"MW_{snap_num:03d}.txt")
m31_file = os.path.join(tmpdir, f"M31_{snap_num:03d}.txt")

if not (os.path.isfile(mw_file) and os.path.isfile(m31_file)):
    continue

try:
    MW_obj  = CenterOfMass(mw_file,  PTYPE)
    M31_obj = CenterOfMass(m31_file, PTYPE)
except Exception as exc:
    print(f"  [WARN] snap {snap_num}: {exc}")
    continue

# Joint COM position (for centring).
x_all = np.concatenate((MW_obj.x,  M31_obj.x))
y_all = np.concatenate((MW_obj.y,  M31_obj.y))
z_all = np.concatenate((MW_obj.z,  M31_obj.z))
m_raw = np.concatenate((MW_obj.m,  M31_obj.m))
xc, yc, zc = MW_obj.COMdefine(x_all, y_all, z_all, m_raw)

# Individual galaxy COMs.
mw_x,  mw_y,  mw_z  = MW_obj.COMdefine( MW_obj.x,  MW_obj.y,  MW_obj.z,  MW_obj.m)
m31_x, m31_y, m31_z = M31_obj.COMdefine(M31_obj.x, M31_obj.y, M31_obj.z, M31_obj.m)

com_mw_pos [i] = [mw_x  - xc, mw_y  - yc, mw_z  - zc]
com_m31_pos[i] = [m31_x - xc, m31_y - yc, m31_z - zc]
separation [i] = np.linalg.norm(com_m31_pos[i] - com_mw_pos[i])

if (i + 1) % 200 == 0:
    print(f"  snap {snap_num:04d}  d = {separation[i]:.1f} kpc")
```

print(f”  COM trajectories done in {time.perf_counter()-t0:.0f}s”)

# ── Relative velocity via finite difference ────────────────────────────────────

# d(separation)/dt — positive = separating, negative = approaching.

# Units: kpc per snapshot index.

valid_sep = np.isfinite(separation)
v_rel     = np.full(ns, np.nan)
if valid_sep.sum() > 2:
v_rel = np.gradient(np.where(valid_sep, separation, np.nan))

# ── Smooth separation for peak detection ──────────────────────────────────────

sep_finite  = np.where(valid_sep, separation, np.nanmedian(separation))
try:
sep_smooth = savgol_filter(sep_finite, SGF_WINDOW, SGF_POLY)
except Exception:
sep_smooth = sep_finite.copy()
sep_smooth[~valid_sep] = np.nan

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §24.2 — EVENT DETECTION                                                   ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Pericentre passages: local minima in d(t) ─────────────────────────────────

# We invert the separation array to find minima as peaks in −d(t).

sep_for_peaks = np.where(np.isfinite(sep_smooth), sep_smooth, np.nanmax(sep_smooth))

peri_indices, peri_props = find_peaks(
-sep_for_peaks,
prominence=PERI_PROMINENCE,
distance=PERI_MIN_SEP_SNAPS,
)

# ── Apocentre passages: local maxima in d(t) ──────────────────────────────────

apo_indices, apo_props = find_peaks(
sep_for_peaks,
prominence=APO_PROMINENCE,
distance=PERI_MIN_SEP_SNAPS,
)

# ── Coalescence epoch: first snapshot where d < D_COAL_KPC permanently ────────

coal_idx = None
for i in range(ns - 1):
if np.isfinite(separation[i]) and separation[i] < D_COAL_KPC:
# Check it stays below threshold for at least 10 subsequent snapshots.
future = separation[i:min(i+10, ns)]
if np.all(future[np.isfinite(future)] < D_COAL_KPC * 2.0):
coal_idx = i
break

# ── Virialisation epoch: from Jeans score (§15 of kinematics pipeline) ────────

virial_idx = None
try:
# jeans_score is defined in the kinematics pipeline, §15.
finite_j = np.isfinite(jeans_score)
if finite_j.sum() > 5:
# Find the last snapshot where score exceeds threshold.
above_thresh = np.where(finite_j & (jeans_score > JEANS_VIRIAL_THRESH))[0]
if len(above_thresh) > 0:
virial_idx = int(above_thresh[-1])
except NameError:
pass   # jeans_score not in scope — skip virialisation detection

# ── Build event catalogue ─────────────────────────────────────────────────────

events = []

for rank, pi in enumerate(peri_indices):
t_e  = time_arr[pi] if np.isfinite(time_arr[pi]) else float(SNAPSHOTS[pi])
events.append({
“type”:   “pericentre”,
“rank”:   rank + 1,
“snap”:   int(SNAPSHOTS[pi]),
“snap_idx”: int(pi),
“time”:   t_e,
“d_kpc”:  float(separation[pi]) if np.isfinite(separation[pi]) else np.nan,
“v_rel”:  float(abs(v_rel[pi]))  if np.isfinite(v_rel[pi])     else np.nan,
})

for rank, ai in enumerate(apo_indices):
t_e  = time_arr[ai] if np.isfinite(time_arr[ai]) else float(SNAPSHOTS[ai])
events.append({
“type”:   “apocentre”,
“rank”:   rank + 1,
“snap”:   int(SNAPSHOTS[ai]),
“snap_idx”: int(ai),
“time”:   t_e,
“d_kpc”:  float(separation[ai]) if np.isfinite(separation[ai]) else np.nan,
“v_rel”:  float(abs(v_rel[ai]))  if np.isfinite(v_rel[ai])     else np.nan,
})

if coal_idx is not None:
t_coal = time_arr[coal_idx] if np.isfinite(time_arr[coal_idx]) else float(SNAPSHOTS[coal_idx])
events.append({
“type”:     “coalescence”,
“rank”:     1,
“snap”:     int(SNAPSHOTS[coal_idx]),
“snap_idx”: int(coal_idx),
“time”:     t_coal,
“d_kpc”:    float(separation[coal_idx]) if np.isfinite(separation[coal_idx]) else np.nan,
“v_rel”:    float(abs(v_rel[coal_idx]))  if np.isfinite(v_rel[coal_idx])     else np.nan,
})

if virial_idx is not None:
t_vir = time_arr[virial_idx] if np.isfinite(time_arr[virial_idx]) else float(SNAPSHOTS[virial_idx])
events.append({
“type”:     “virialisation”,
“rank”:     1,
“snap”:     int(SNAPSHOTS[virial_idx]),
“snap_idx”: int(virial_idx),
“time”:     t_vir,
“d_kpc”:    float(separation[virial_idx]) if np.isfinite(separation[virial_idx]) else np.nan,
“v_rel”:    np.nan,
})

# Sort by time.

events.sort(key=lambda e: e[“time”] if np.isfinite(e[“time”]) else 1e30)

print(f”\n  Detected {len(peri_indices)} pericentre(s), “
f”{len(apo_indices)} apocentre(s), “
f”coalescence={‘yes’ if coal_idx is not None else ‘not detected’}”)
for ev in events:
print(f”    {ev[‘type’]:14s}  snap={ev[‘snap’]:04d}  “
f”t={ev[‘time’]:.3f}  d={ev[‘d_kpc’]:.1f} kpc”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §24.3 — ORBITAL ENERGY AND ANGULAR MOMENTUM                              ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# 

# Specific orbital energy:

# E_orb = ½ v_rel² − G (M_MW + M_M31) / d

# 

# Specific orbital angular momentum:

# L_orb = |r_rel × v_rel|

# 

# Both should decrease over time as dynamical friction drains the orbit.

# Comparing the rate of decrease to theoretical predictions (Chandrasekhar

# formula) provides a check on the simulation’s treatment of dynamical friction.

# Total system mass (from the first valid snapshot).

M_total_system = np.nan
for i in range(ns):
snap_num = SNAPSHOTS[i]
mw_file  = os.path.join(tmpdir, f”MW_{snap_num:03d}.txt”)
m31_file = os.path.join(tmpdir, f”M31_{snap_num:03d}.txt”)
if os.path.isfile(mw_file) and os.path.isfile(m31_file):
try:
snap_data      = load_snapshot_particles(mw_file, m31_file)
M_total_system = snap_data[“m_msun”].sum()
break
except Exception:
pass

# Relative position and velocity vectors.

r_rel = com_m31_pos - com_mw_pos      # (ns, 3)  [kpc]
v_rel_vec = np.gradient(r_rel, axis=0)  # (ns, 3)  [kpc/snap] — convert to km/s below

# Relative speed.

v_rel_mag = np.linalg.norm(v_rel_vec, axis=1)   # [kpc/snap]

# Orbital energy (using raw snapshot units for now; can rescale with Δt).

E_orb = np.full(ns, np.nan)
L_orb = np.full(ns, np.nan)

if np.isfinite(M_total_system):
for i in range(ns):
if not (np.isfinite(separation[i]) and separation[i] > 0):
continue
v2      = np.sum(v_rel_vec[i]**2)
E_orb[i] = 0.5 * v2 - G_KPC_KMS2_MSUN * M_total_system / separation[i]
j_vec   = np.cross(r_rel[i], v_rel_vec[i])
L_orb[i] = np.linalg.norm(j_vec)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §24.4 — FIGURES                                                           ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

BG    = “#0d0d18”
MUTED = “#7070a0”

# Event colours and markers.

EV_STYLE = {
“pericentre”:   {“color”: “#ff5566”, “marker”: “v”, “ls”: “–”},
“apocentre”:    {“color”: “#4a8fff”, “marker”: “^”, “ls”: “:”},
“coalescence”:  {“color”: “#ffcc44”, “marker”: “*”, “ls”: “-”},
“virialisation”:{“color”: “#00d4aa”, “marker”: “D”, “ls”: “-.”},
}

def _ax(ax, xlabel=””, ylabel=””, title=””, log_x=False, log_y=False):
ax.set_facecolor(BG)
for sp in ax.spines.values():
sp.set_edgecolor(”#2a2a4a”)
ax.tick_params(colors=”#9090b0”, labelsize=8)
ax.set_xlabel(xlabel, fontsize=9,  color=”#c8c8e8”)
ax.set_ylabel(ylabel, fontsize=9,  color=”#c8c8e8”)
ax.set_title(title,   fontsize=10, color=”#c8c8e8”, pad=5)
if log_x: ax.set_xscale(“log”)
if log_y: ax.set_yscale(“log”)
return ax

def _annotate_events(ax, events, ypos_frac=0.92, skip_types=None):
“”“Add vertical lines and labels for all detected events.”””
skip_types = skip_types or []
ylims = ax.get_ylim()
ypos  = ylims[0] + ypos_frac * (ylims[1] - ylims[0])
for ev in events:
if ev[“type”] in skip_types:
continue
st = EV_STYLE.get(ev[“type”], {“color”: “#ffffff”, “ls”: “–”})
ax.axvline(ev[“time”], color=st[“color”], lw=0.9, ls=st[“ls”], alpha=0.7)
label = f”P{ev[‘rank’]}” if ev[“type”] == “pericentre” else   
f”A{ev[‘rank’]}” if ev[“type”] == “apocentre”  else   
ev[“type”][:4].upper()
ax.text(ev[“time”], ypos, label, color=st[“color”],
fontsize=6, ha=“center”, va=“top”,
bbox=dict(boxstyle=“round,pad=0.15”, fc=BG, ec=“none”, alpha=0.6))

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 1 — SEPARATION d(t) WITH FULL EVENT ANNOTATION

# ══════════════════════════════════════════════════════════════════════════════

print(”\n[Fig 1]  Separation with event annotations …”)

fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(13, 8), facecolor=BG,
sharex=True, gridspec_kw={“hspace”: 0.08})

_ax(ax1a, ylabel=“Separation [kpc]”,
title=“MW–M31 Orbital Separation with Merger Events”)
ax1a.plot(time_arr, separation,  color=”#4a8fff”, lw=1.2, alpha=0.4,
label=“Raw d(t)”)
ax1a.plot(time_arr, sep_smooth,  color=”#4a8fff”, lw=2.2,
label=“Smoothed d(t)”)
ax1a.axhline(D_COAL_KPC, color=”#ffcc44”, lw=0.8, ls=”–”, alpha=0.5,
label=f”Coalescence threshold {D_COAL_KPC:.0f} kpc”)
ax1a.set_yscale(“log”)
_annotate_events(ax1a, events)
ax1a.legend(fontsize=7, loc=“upper right”)

*ax(ax1b, xlabel=time_label,
ylabel=r”$v*{\rm rel}$ [kpc snap$^{-1}$]”,
title=“Relative Approach / Recession Speed”)
ax1b.plot(time_arr, v_rel, color=”#e8673a”, lw=1.6)
ax1b.axhline(0, color=”#555577”, lw=0.8, ls=”–”)
ax1b.fill_between(time_arr,
np.where(np.isfinite(v_rel) & (v_rel < 0), v_rel, 0),
alpha=0.15, color=”#e8673a”, label=“Approaching”)
ax1b.fill_between(time_arr,
np.where(np.isfinite(v_rel) & (v_rel > 0), v_rel, 0),
alpha=0.12, color=”#4a8fff”, label=“Receding”)
_annotate_events(ax1b, events, ypos_frac=0.85)
ax1b.legend(fontsize=7)

# Legend patches for event types.

legend_patches = [
mpatches.Patch(color=EV_STYLE[k][“color”], label=k.capitalize())
for k in [“pericentre”, “apocentre”, “coalescence”, “virialisation”]
if any(e[“type”] == k for e in events)
]
if legend_patches:
ax1a.legend(handles=legend_patches + ax1a.get_legend_handles_labels()[0],
fontsize=7, loc=“upper right”)

fig1.savefig(os.path.join(OUT_DIR, “section24_separation.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig1)
print(”  Saved: section24_separation.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 2 — ORBITAL PHASE DIAGRAM (d, v_rel)

# ══════════════════════════════════════════════════════════════════════════════

# 

# Plotting v_rel vs. d traces out the orbital trajectory in phase space.

# A bound decaying orbit appears as an inward-spiralling loop.  Each loop

# corresponds to one pericentre passage.  The spiral terminates at the

# coalescence point (small d, small v_rel).

print(”[Fig 2]  Orbital phase diagram …”)

fig2, ax2 = plt.subplots(figsize=(8, 7), facecolor=BG)
*ax(ax2, xlabel=“Separation d [kpc]”,
ylabel=r”$|v*{\rm rel}|$ [kpc snap$^{-1}$]”,
title=“Orbital Phase Diagram  $(d,,|v_{\rm rel}|)$”,
log_x=True)

# Colour the trajectory by time.

valid_pd = np.isfinite(separation) & np.isfinite(v_rel)
t_norm   = (time_arr - np.nanmin(time_arr)) /   
(np.nanmax(time_arr) - np.nanmin(time_arr) + 1e-30)

sc = ax2.scatter(separation[valid_pd], np.abs(v_rel[valid_pd]),
c=t_norm[valid_pd], cmap=“plasma”,
s=4, alpha=0.7, rasterized=True)
fig2.colorbar(sc, ax=ax2, label=time_label, pad=0.01)

# Mark pericentre and apocentre points.

for ev in events:
si = ev[“snap_idx”]
if not (np.isfinite(separation[si]) and np.isfinite(v_rel[si])):
continue
st = EV_STYLE.get(ev[“type”], {“color”: “#ffffff”, “marker”: “o”})
ax2.scatter(separation[si], abs(v_rel[si]),
color=st[“color”], marker=st[“marker”],
s=120, zorder=5, edgecolors=“white”, linewidths=0.5)

# Add event type legend.

handles = [
Line2D([0], [0], marker=EV_STYLE[k][“marker”], color=“w”,
markerfacecolor=EV_STYLE[k][“color”], markersize=8, label=k.capitalize())
for k in [“pericentre”, “apocentre”, “coalescence”, “virialisation”]
if any(e[“type”] == k for e in events)
]
if handles:
ax2.legend(handles=handles, fontsize=8)

fig2.savefig(os.path.join(OUT_DIR, “section24_phase_diagram.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig2)
print(”  Saved: section24_phase_diagram.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 3 — ORBITAL ENERGY AND ANGULAR MOMENTUM VS. TIME

# ══════════════════════════════════════════════════════════════════════════════

# 

# E_orb should decrease monotonically as dynamical friction dissipates the

# orbit.  Sudden jumps mark close passages where the point-mass approximation

# breaks down and the two galaxy potential wells overlap.

# L_orb should also decrease, with each pericentre passage transferring

# angular momentum into internal degrees of freedom (halo spin, tidal streams).

print(”[Fig 3]  Orbital energy and angular momentum …”)

fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(11, 8), facecolor=BG,
sharex=True, gridspec_kw={“hspace”: 0.08})

valid_E = np.isfinite(E_orb)
valid_L = np.isfinite(L_orb)

*ax(ax3a, ylabel=r”$E*{\rm orb}$ [kpc$^2$ snap$^{-2}$]”,
title=“Specific Orbital Energy and Angular Momentum”)
if valid_E.any():
ax3a.plot(time_arr[valid_E], E_orb[valid_E], color=”#ff9944”, lw=1.8,
label=r”$E_{\rm orb} = \frac{1}{2}v_{\rm rel}^2 - GM/d$”)
ax3a.axhline(0, color=”#555577”, lw=0.7, ls=”–”)
ax3a.text(np.nanmin(time_arr[valid_E]) * 1.01, 0.02 * np.nanmax(abs(E_orb[valid_E])),
“bound  (E < 0)”, color=MUTED, fontsize=7)
_annotate_events(ax3a, events, ypos_frac=0.88)
ax3a.legend(fontsize=8)

*ax(ax3b, xlabel=time_label,
ylabel=r”$L*{\rm orb}$ [kpc$^2$ snap$^{-1}$]”,
title=“Specific Orbital Angular Momentum”)
if valid_L.any():
ax3b.plot(time_arr[valid_L], L_orb[valid_L], color=”#aa55ff”, lw=1.8,
label=r”$L_{\rm orb} = |r_{\rm rel} \times v_{\rm rel}|$”)
_annotate_events(ax3b, events, ypos_frac=0.88)
ax3b.legend(fontsize=8)

fig3.savefig(os.path.join(OUT_DIR, “section24_orbital_energy.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig3)
print(”  Saved: section24_orbital_energy.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 4 — EVENT TABLE FIGURE

# ══════════════════════════════════════════════════════════════════════════════

print(”[Fig 4]  Event table …”)

fig4, ax4 = plt.subplots(figsize=(12, max(3.0, len(events) * 0.55 + 1.5)),
facecolor=BG)
ax4.set_facecolor(BG)
ax4.axis(“off”)

col_headers = [“Event”, “Snap”, “Time”, “d [kpc]”,
“|v_rel|”, “Type”]
rows = []
for ev in events:
t_str  = f”{ev[‘time’]:.3f}” if np.isfinite(ev[‘time’])  else “—”
d_str  = f”{ev[‘d_kpc’]:.1f}”  if np.isfinite(ev[‘d_kpc’]) else “—”
v_str  = f”{ev[‘v_rel’]:.2f}”  if np.isfinite(ev[‘v_rel’]) else “—”
label  = f”{ev[‘type’].capitalize()} #{ev[‘rank’]}”   
if ev[‘rank’] > 0 else ev[‘type’].capitalize()
rows.append([label, str(ev[‘snap’]), t_str, d_str, v_str, ev[‘type’]])

if not rows:
rows = [[“No events detected”, “”, “”, “”, “”, “”]]

tbl = ax4.table(
cellText=[[r[i] for i in range(5)] for r in rows],
colLabels=col_headers[:5],
loc=“center”,
cellLoc=“center”,
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.3, 1.8)

for (r, c), cell in tbl.get_celld().items():
if r == 0:
cell.set_facecolor(”#1a1a3a”)
cell.set_text_props(color=”#c8c8e8”, fontweight=“bold”)
else:
ev_type = rows[r - 1][5]
row_color = EV_STYLE.get(ev_type, {}).get(“color”, “#ffffff”)
cell.set_facecolor(”#0d0d18” if r % 2 == 0 else “#141428”)
cell.set_text_props(color=row_color if c == 0 else “#c8c8e8”)
cell.set_edgecolor(”#2a2a4a”)

ax4.set_title(“Merger Event Catalogue”, fontsize=11, color=”#c8c8e8”, pad=12)

fig4.savefig(os.path.join(OUT_DIR, “section24_events_table.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig4)
print(”  Saved: section24_events_table.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 5 — HORIZONTAL TIMELINE BANNER

# ══════════════════════════════════════════════════════════════════════════════

# 

# A horizontal timeline figure suitable for a paper methods section or poster.

# The timeline runs left to right in time units; events are marked as

# coloured vertical ticks with labels above and below alternating to avoid

# overlap.

print(”[Fig 5]  Timeline banner …”)

fig5, ax5 = plt.subplots(figsize=(14, 3.5), facecolor=BG)
ax5.set_facecolor(BG)
ax5.spines[“top”].set_visible(False)
ax5.spines[“right”].set_visible(False)
ax5.spines[“left”].set_visible(False)
ax5.spines[“bottom”].set_edgecolor(”#4a8fff”)
ax5.tick_params(left=False, labelleft=False, colors=”#9090b0”, labelsize=8)
ax5.set_yticks([])

t_valid = time_arr[np.isfinite(time_arr)]
if len(t_valid) > 0:
ax5.set_xlim(t_valid.min() - 0.1, t_valid.max() + 0.1)
ax5.set_ylim(-1.5, 1.5)
ax5.set_xlabel(time_label, fontsize=9, color=”#c8c8e8”)
ax5.set_title(“MW–M31 Merger Timeline”, fontsize=11, color=”#c8c8e8”, pad=8)

# Draw the baseline.

if len(t_valid) > 0:
ax5.axhline(0, color=”#4a8fff”, lw=1.5, xmin=0.01, xmax=0.99, zorder=1)

for k, ev in enumerate(events):
st    = EV_STYLE.get(ev[“type”], {“color”: “#ffffff”})
t_ev  = ev[“time”]
if not np.isfinite(t_ev):
continue

```
# Alternating label positions: odd events above, even below.
y_tick  = 0.6 if k % 2 == 0 else -0.6
y_label = 1.1 if k % 2 == 0 else -1.1
va      = "bottom" if k % 2 == 0 else "top"

ax5.annotate("",
             xy=(t_ev, 0), xytext=(t_ev, y_tick),
             arrowprops=dict(arrowstyle="-", color=st["color"], lw=1.5))
ax5.scatter([t_ev], [0], color=st["color"],
            marker=st.get("marker", "o"), s=60, zorder=5)

label = f"{ev['type'].capitalize()}"
if ev["type"] in ("pericentre", "apocentre"):
    label += f" #{ev['rank']}\n{ev['d_kpc']:.0f} kpc"
elif ev["type"] == "coalescence":
    label += f"\nd < {D_COAL_KPC:.0f} kpc"

ax5.text(t_ev, y_label, label, ha="center", va=va,
         fontsize=6.5, color=st["color"],
         bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec="none", alpha=0.7))
```

fig5.savefig(os.path.join(OUT_DIR, “section24_timeline_banner.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig5)
print(”  Saved: section24_timeline_banner.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 6 — MASTER SUMMARY PANEL

# ══════════════════════════════════════════════════════════════════════════════

print(”\n[Summary]  Master summary panel …”)

fig6 = plt.figure(figsize=(16, 10), facecolor=BG)
gs6  = gridspec.GridSpec(2, 2, figure=fig6,
hspace=0.38, wspace=0.32,
left=0.08, right=0.97,
top=0.93, bottom=0.07)

# (0,0) Separation.

ax_s00 = fig6.add_subplot(gs6[0, 0])
_ax(ax_s00, xlabel=time_label, ylabel=“d [kpc]”,
title=“Galaxy Separation”, log_y=True)
ax_s00.plot(time_arr, sep_smooth, color=”#4a8fff”, lw=2.0)
ax_s00.axhline(D_COAL_KPC, color=”#ffcc44”, lw=0.7, ls=”–”, alpha=0.5)
_annotate_events(ax_s00, events, ypos_frac=0.88)

# (0,1) Phase diagram.

ax_s01 = fig6.add_subplot(gs6[0, 1])
*ax(ax_s01, xlabel=“d [kpc]”, ylabel=r”$|v*{\rm rel}|$”,
title=“Orbital Phase Diagram”, log_x=True)
if valid_pd.any():
ax_s01.scatter(separation[valid_pd], np.abs(v_rel[valid_pd]),
c=t_norm[valid_pd], cmap=“plasma”, s=3, alpha=0.6, rasterized=True)
for ev in events:
si = ev[“snap_idx”]
if np.isfinite(separation[si]) and np.isfinite(v_rel[si]):
st = EV_STYLE.get(ev[“type”], {“color”: “#ffffff”, “marker”: “o”})
ax_s01.scatter(separation[si], abs(v_rel[si]),
color=st[“color”], marker=st[“marker”], s=80, zorder=5)

# (1,0) Orbital energy.

ax_s10 = fig6.add_subplot(gs6[1, 0])
*ax(ax_s10, xlabel=time_label, ylabel=r”$E*{\rm orb}$”,
title=“Specific Orbital Energy”)
if valid_E.any():
ax_s10.plot(time_arr[valid_E], E_orb[valid_E], color=”#ff9944”, lw=1.8)
ax_s10.axhline(0, color=”#555577”, lw=0.6, ls=”–”)
_annotate_events(ax_s10, events, ypos_frac=0.88)

# (1,1) Angular momentum.

ax_s11 = fig6.add_subplot(gs6[1, 1])
*ax(ax_s11, xlabel=time_label, ylabel=r”$L*{\rm orb}$”,
title=“Specific Orbital Angular Momentum”)
if valid_L.any():
ax_s11.plot(time_arr[valid_L], L_orb[valid_L], color=”#aa55ff”, lw=1.8)
_annotate_events(ax_s11, events, ypos_frac=0.88)

fig6.suptitle(“Section 24 Summary  ·  Merger Timeline & Pericentre Detection”,
fontsize=13, color=”#c8c8e8”, fontweight=“bold”)
fig6.savefig(os.path.join(OUT_DIR, “section24_summary_panel.png”),
dpi=200, bbox_inches=“tight”, facecolor=BG)
plt.close(fig6)
print(”  Saved: section24_summary_panel.png”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §24.5 — SECTION COMPLETE                                                  ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

print(”\n” + “=”*80)
print(”  SECTION 24 COMPLETE”)
print(”=”*80)
outputs_24 = [
“section24_separation.png”,
“section24_phase_diagram.png”,
“section24_orbital_energy.png”,
“section24_events_table.png”,
“section24_timeline_banner.png”,
“section24_summary_panel.png”,
]
for fn in outputs_24:
fp   = os.path.join(OUT_DIR, fn)
size = os.path.getsize(fp) / 1e6 if os.path.isfile(fp) else 0.0
print(f”  {fn:<45} {size:6.2f} MB”)
print(”=”*80)

# ── Print pericentre summary to stdout ────────────────────────────────────────

print(”\n  MERGER EVENT SUMMARY”)
print(f”  {‘Event’:<22} {‘Snap’:>6} {‘Time’:>10} {‘d [kpc]’:>10} {’|v_rel|’:>10}”)
print(f”  {’-’*22} {’-’*6} {’-’*10} {’-’*10} {’-’*10}”)
for ev in events:
t_str = f”{ev[‘time’]:.3f}” if np.isfinite(ev[‘time’]) else “—”
d_str = f”{ev[‘d_kpc’]:.1f}” if np.isfinite(ev[‘d_kpc’]) else “—”
v_str = f”{ev[‘v_rel’]:.2f}” if np.isfinite(ev[‘v_rel’]) else “—”
label = f”{ev[‘type’].capitalize()} #{ev[‘rank’]}”
print(f”  {label:<22} {ev[‘snap’]:>6} {t_str:>10} {d_str:>10} {v_str:>10}”)
print(”=”*80)
