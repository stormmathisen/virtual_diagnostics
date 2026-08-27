"""End-to-end worked example: Cheetah beam in, instrument readings out.

Tracks a bunch through a short lattice in Cheetah, then reads it out on every
diagnostic in the package.  Running this writes the figures and the printed
output used throughout the documentation::

    python examples/clara_diagnostics.py

Requires the ``cheetah`` extra.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import cheetah

import virtual_diagnostics as vd

OUT = "docs/images"
SEED = 2024

# ---------------------------------------------------------------- 1. the beam

source = cheetah.ParticleBeam.from_twiss(
    num_particles=200_000,
    beta_x=torch.tensor(5.0),
    beta_y=torch.tensor(5.0),
    alpha_x=torch.tensor(0.0),
    alpha_y=torch.tensor(0.0),
    emittance_x=torch.tensor(1e-9),
    emittance_y=torch.tensor(1e-9),
    energy=torch.tensor(100e6),
    sigma_tau=torch.tensor(0.3e-3),
    sigma_p=torch.tensor(1e-3),
    total_charge=torch.tensor(250e-12),
)

lattice = cheetah.Segment(
    [
        cheetah.Drift(length=torch.tensor(0.5)),
        cheetah.Quadrupole(length=torch.tensor(0.1), k1=torch.tensor(3.0)),
        cheetah.Drift(length=torch.tensor(1.5)),
    ]
)

beam = vd.from_cheetah(lattice.track(source))
print("beam at the screen:")
print(f"  {beam}")
print(f"  charge          {beam.total_charge * 1e12:8.2f} pC")
print(f"  energy          {beam.mean_energy / 1e6:8.2f} MeV")
print(f"  energy spread   {beam.relative_energy_spread * 100:8.3f} %")
print(f"  sigma_x         {beam.sigma_x * 1e6:8.2f} um")
print(f"  sigma_y         {beam.sigma_y * 1e6:8.2f} um")
print(f"  sigma_t         {beam.sigma_t * 1e15:8.2f} fs")

# ------------------------------------------------------- 2. screen -> an image

screen = vd.Screen(
    pixel_size=12e-6,
    resolution=(640, 480),
    psf_sigma=30e-6,
    counts_per_pc=3e3,
    dark_offset=100.0,
    read_noise=6.0,
    bits=12,
)
image = screen.measure(beam, rng=SEED)

raw = image.beam_size(deconvolved=False)
corrected = image.beam_size()
print("\nscreen (YAG, 12 um pixels, 30 um PSF):")
print(f"  frame           {image.counts.shape[1]} x {image.counts.shape[0]} px, {screen.bits}-bit")
print(f"  peak            {image.counts.max()} counts")
print(f"  off screen      {image.off_screen_fraction * 100:8.3f} %")
print(f"  saturated       {image.saturated_fraction * 100:8.3f} % of pixels")
print(f"  measured sigma  {raw[0] * 1e6:8.2f} / {raw[1] * 1e6:8.2f} um  (raw)")
print(f"  measured sigma  {corrected[0] * 1e6:8.2f} / {corrected[1] * 1e6:8.2f} um  (deconvolved)")
print(f"  true sigma      {beam.sigma_x * 1e6:8.2f} / {beam.sigma_y * 1e6:8.2f} um")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
vd.plot_image(image, ax=axes[0])
axes[0].set_title("YAG screen, single shot")
vd.plot_projections(image, ax=axes[1])
axes[1].set_title("projections")
fig.tight_layout()
fig.savefig(f"{OUT}/screen.png", dpi=110)

# ------------------------------------------- 3. current monitors -> a waveform

ict = vd.CurrentMonitor(
    rise_time=20e-9,
    droop_time=5e-6,
    transimpedance=1.25,
    noise=2e-4,
    sample_rate=2e9,
    duration=1e-6,
    readout=vd.LogAmpReadout(qcal=1e-12, ucal=0.5),
)
# A 100 ps rise time is far longer than this 1 ps bunch, so an FCT integrates
# it just as the ICT does --- it simply has a much shorter impulse response.
# Only a pickup faster than the bunch follows the instantaneous current.
fct = vd.CurrentMonitor(
    rise_time=100e-12,
    droop_time=None,
    transimpedance=1.0,
    noise=1e-4,
    sample_rate=2e12,
    duration=1e-9,
)
fast = vd.CurrentMonitor(
    rise_time=200e-15,
    droop_time=None,
    transimpedance=1.0,
    noise=1e-3,
    sample_rate=2e13,
    duration=20e-12,
)

t_ict, v_ict = ict.measure(beam, rng=SEED)
t_fct, v_fct = fct.measure(beam, rng=SEED)
t_fast, v_fast = fast.measure(beam, rng=SEED)

print("\ncurrent monitors:")
print(f"  ICT peak        {v_ict.max() * 1e3:8.3f} mV")
print(f"  ICT integral    {ict.integrated_charge(t_ict, v_ict) * 1e12:8.2f} pC  (droop under-reads)")
print(f"  ICT bandwidth   {ict.bandwidth / 1e6:8.2f} MHz")
print(f"  FCT peak        {v_fct.max():8.2f} A   (100 ps rise: still integrating)")
print(f"  FCT integral    {fct.integrated_charge(t_fct, v_fct) * 1e12:8.2f} pC")
print(f"  fast peak       {v_fast.max():8.2f} A   (200 fs rise: follows the current)")
print(f"  peak current    {beam.total_charge / (np.sqrt(2 * np.pi) * beam.sigma_t):8.2f} A   (true)")
print(f"  true charge     {beam.total_charge * 1e12:8.2f} pC")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
vd.plot_signal(t_ict * 1e0, v_ict, ax=axes[0], time_units="ns", label="ICT")
axes[0].set_title("ICT: 20 ns rise, 5 us droop")
vd.plot_signal(t_fast, v_fast, ax=axes[1], time_units="ps", label="200 fs pickup")
axes[1].set_title("fast pickup: follows the beam current")
fig.tight_layout()
fig.savefig(f"{OUT}/current_monitors.png", dpi=110)

vd.to_spice_pwl("examples/ict_pulse.pwl", t_ict, v_ict)
print("  wrote examples/ict_pulse.pwl for SPICE")

# ------------------------------------------------------ 4. BPMs

bpm = vd.BPM(resolution=10e-6, reference_charge=100e-12, offset=(30e-6, 0.0))
print("\nBPM, readout model (10 um resolution at 100 pC, 30 um electrical offset):")
print("   charge     reading x     scatter    predicted")
for charge in (250e-12, 50e-12, 5e-12):
    shots = [bpm.measure(beam.scaled_charge(charge), rng=s).x for s in range(200)]
    print(
        f"  {charge * 1e12:6.0f} pC  {np.mean(shots) * 1e6:8.2f} um  "
        f"{np.std(shots) * 1e6:8.2f} um  {bpm.noise_at(charge) * 1e6:8.2f} um"
    )

button = vd.ButtonBPM(pipe_radius=20e-3)
button.sensitivity = button.calibrate_sensitivity()
stripline = vd.StriplineBPM(pipe_radius=20e-3, length=100e-3)
stripline.sensitivity = stripline.calibrate_sensitivity()

print("\nper-electrode models:")
print(f"  button electrode R*C     {button.electrode.droop_time * 1e12:8.1f} ps")
print(f"  stripline round trip     {stripline.round_trip_time() * 1e12:8.1f} ps  (2L/c)")
print(f"  stripline peak response  {stripline.quarter_wave_frequency / 1e9:8.3f} GHz  (c/4L)")
print("    x (mm)   button reads   stripline reads")
for x0 in (0.0, 1e-3, 3e-3, 6e-3, 12e-3):
    offset_beam = beam.shifted(dx=x0 - beam.centroid_x)
    xb = button.position(button.measure(offset_beam, rng=SEED))[0]
    xs = stripline.position(stripline.measure(offset_beam, rng=SEED))[0]
    print(f"    {x0 * 1e3:6.2f} {xb * 1e3:14.4f} {xs * 1e3:17.4f}")
print("  (both read low near the wall: the Poisson kernel, not a bug)")

button_signals = button.measure(beam.shifted(dx=3e-3 - beam.centroid_x), rng=SEED)
stripline_signals = stripline.measure(beam.shifted(dx=3e-3 - beam.centroid_x), rng=SEED)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
# Electrodes 1 (beam right) and 2 (beam left) with the beam 3 mm to the right:
# the amplitude difference between them is the position signal.
for ax, signals, title, span in (
    (axes[0], button_signals, "button: R*C differentiated", 1.0e-9),
    (axes[1], stripline_signals, "stripline: in-pulse, out-pulse", 1.2e-9),
):
    for index, label in ((0, "right (+45 deg)"), (1, "left (+135 deg)")):
        vd.plot_signal(signals.t, signals.volts[index], ax=ax, time_units="ns", label=label)
    ax.set_xlim(-0.2, span * 1e9)
    ax.set_title(title)
fig.tight_layout()
fig.savefig(f"{OUT}/bpm_electrodes.png", dpi=110)

# --------------------------------- 4b. signal conditioning in SPICE

if vd.ngspice_available():
    combiner = vd.SpiceFrontEnd(
        netlist="examples/netlists/delay_line_combiner.cir",
        sources=("Vb1", "Vb2", "Vb3", "Vb4"),
        outputs="xout",
    )

    def peaks(t, v, delay=2e-9, half=0.5e-9):
        lead = t[np.flatnonzero(v >= 0.2 * v.max())[0]]
        return (
            v[np.abs(t - lead) <= half].max(),
            v[np.abs(t - (lead + delay)) <= half].max(),
        )

    print("\ndelay-line combiner (the delay line lives in the netlist):")
    print("    x (mm)   direct (V)  delayed (V)   difference/sum")
    offsets, ratios, traces = [], [], {}
    for x0 in (-2e-3, -1e-3, 0.0, 1e-3, 2e-3):
        signals = button.measure(beam.shifted(dx=x0 - beam.centroid_x), rng=SEED)
        t_out, trace = combiner.run(signals.t, signals.volts)
        direct, delayed = peaks(t_out, trace)
        ratio = (direct - delayed) / (direct + delayed)
        offsets.append(x0)
        ratios.append(ratio)
        traces[x0] = (t_out, trace)
        print(f"    {x0 * 1e3:6.2f} {direct:12.4f} {delayed:12.4f} {ratio:16.6f}")

    slope, intercept = np.polyfit(ratios, offsets, 1)
    print(f"  calibration: x = {slope * 1e3:.3f} mm * (d/s) + {intercept * 1e6:.1f} um")
    print(f"  geometric expectation b/sqrt(2) = {20e-3 / np.sqrt(2) * 1e3:.3f} mm")
    print("  the offset is the branch asymmetry -- one side goes through the line")

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for x0 in (-2e-3, 0.0, 2e-3):
        t_out, trace = traces[x0]
        vd.plot_signal(t_out, trace, ax=ax, time_units="ns", label=f"x = {x0 * 1e3:+.0f} mm")
    ax.set_title("delay-line combiner output")
    fig.tight_layout()
    fig.savefig(f"{OUT}/delay_line.png", dpi=110)
else:
    print("\ndelay-line combiner: skipped, ngspice not available")

# -------------------------------------- 5. spectrometer and transverse deflector

dispersion = 0.4
dispersed = beam.shifted()
dispersed.x = dispersed.x + dispersion * (beam.energies / beam.mean_energy - 1.0)
spectrometer = vd.Spectrometer(
    screen=vd.Screen(pixel_size=20e-6, resolution=(400, 200), counts_per_pc=3e3),
    dispersion=dispersion,
    reference_energy=beam.mean_energy,
)
spec_image = spectrometer.measure(dispersed, rng=SEED)
measured_spread = spectrometer.energy_spread(spec_image, beam_size=beam.sigma_x)
print("\nspectrometer (0.4 m dispersion):")
print(f"  mean energy     {spectrometer.mean_energy(spec_image) / 1e6:8.3f} MeV")
print(f"  energy spread   {measured_spread / 1e3:8.2f} keV  (measured)")
print(f"  true spread     {beam.energy_spread / 1e3:8.2f} keV")

shear = 3e8
streaked = beam.shifted()
streaked.y = streaked.y + shear * beam.t
tds = vd.StreakedScreen(
    screen=vd.Screen(pixel_size=20e-6, resolution=(200, 400), counts_per_pc=3e3),
    shear=shear,
)
tds_image = tds.measure(streaked, rng=SEED)
measured_length = tds.bunch_length(tds_image, unstreaked_size=beam.sigma_y)
print(f"\ntransverse deflector ({shear:.1e} m/s shear):")
print(f"  bunch length    {measured_length * 1e15:8.2f} fs  (measured)")
print(f"  true length     {beam.sigma_t * 1e15:8.2f} fs")
print(f"  pixel limit     {tds.resolution * 1e15:8.2f} fs")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
energy, intensity = spectrometer.spectrum(spec_image)
axes[0].plot(energy / 1e6, intensity)
axes[0].set_xlabel("energy (MeV)")
axes[0].set_ylabel("intensity (counts)")
axes[0].set_title("spectrometer projection")
time, profile = tds.profile(tds_image)
axes[1].plot(time * 1e15, profile)
axes[1].set_xlabel("time (fs)")
axes[1].set_ylabel("intensity (counts)")
axes[1].set_title("streaked longitudinal profile")
fig.tight_layout()
fig.savefig(f"{OUT}/spectrometer_tds.png", dpi=110)

# ------------------------------------------ 6. coherent radiation compression scan

crm = vd.CoherentRadiationMonitor(band=(0.3e12, 3e12), calibration=1e18, noise=1e-6)
lengths = np.array([1500e-15, 1000e-15, 600e-15, 400e-15, 250e-15, 150e-15])
signals = []
print("\nCTR monitor (0.3-3 THz band):")
for sigma_t in lengths:
    compressed = beam.shifted()
    compressed.t = beam.t * (sigma_t / beam.sigma_t)
    signals.append(crm.measure(compressed, rng=SEED))
    print(f"  sigma_t {sigma_t * 1e15:6.0f} fs -> {signals[-1] * 1e3:10.4f} mV")

fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.semilogy(lengths * 1e15, np.array(signals) * 1e3, "o-")
ax.set_xlabel("bunch length (fs)")
ax.set_ylabel("CTR signal (mV)")
ax.set_title("compression scan")
fig.tight_layout()
fig.savefig(f"{OUT}/ctr_scan.png", dpi=110)

print(f"\nfigures written to {OUT}/")
