"""The CLARA analog front end, driven by a Faraday cup and a wall current monitor.

Reproduces the signal chain of

    S. L. Mathisen, T. H. Pacey, R. J. Smith, "Analog Front End for Measuring
    1 to 250 pC Bunch Charge at CLARA", IBIC2022, MOP32.
    doi:10.18429/JACoW-IBIC2022-MOP32

as a SPICE netlist (``netlists/clara_front_end.cir``), and drives it with two
very different charge devices:

* a **Faraday cup**, DC coupled, its bandwidth set by the impedance it
  discharges through;
* a **wall current monitor**, bandpass from 100 kHz to 6 GHz.

The paper's point is that the front end is deliberately agnostic to which one is
connected: a 100 MHz input filter throws away the difference, and the charge
integrator returns a peak proportional to bunch charge either way. Only the
polarity differs, because a WCM measures the beam's image current (Figs. 4, 5).

Run::

    python examples/clara_front_end.py

Requires the ngspice binary. See docs/guides/spice.md.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import virtual_diagnostics as vd

OUT = "docs/images"
NETLIST = "examples/netlists/clara_front_end.cir"
SEED = 2024

# Table 1 of the paper: the five switchable feedback capacitances and the range
# each is specified over. RF is set for RF * CF = 1 us so the stage integrates
# the bunch fully and the peak is -GOUT * Q / CF (see the netlist header).
# GOUT is fitted to the saturation points of Fig. 3; the paper does not give the
# per-path gains, and they are not a common factor.
SETTINGS = {
    #  name        CF        RF       GOUT   range (pC)
    "highest": (2e-12, 500e3, 0.23, (0, 10)),
    "high": (15e-12, 66.7e3, 1.02, (2, 20)),
    "medium": (51e-12, 19.6e3, 1.90, (2, 40)),
    "low": (300e-12, 3.33e3, 2.20, (10, 150)),
    "lowest": (1e-9, 1.00e3, 2.80, (10, 250)),
}

FULL_SCALE = 1.5  # volts; where Fig. 3 shows the outputs flattening off


def front_end(setting: str, max_points: int = 8000) -> vd.SpiceFrontEnd:
    """The front end configured for one sensitivity setting."""
    cf, rf, gout, _ = SETTINGS[setting]
    netlist = open(NETLIST).read()
    netlist = netlist.replace(".param CF=1n", f".param CF={cf:.6e}")
    netlist = netlist.replace(".param RF=1k", f".param RF={rf:.6e}")
    netlist = netlist.replace(".param GOUT=2.8", f".param GOUT={gout:.6e}")
    return vd.SpiceFrontEnd(
        netlist=netlist, sources="Vin", outputs="out", max_points=max_points
    )


def bunch(charge: float, n: int = 50_000) -> vd.Beam:
    """A short CLARA-like bunch carrying a given charge."""
    rng = np.random.default_rng(SEED)
    return vd.Beam(
        x=rng.normal(0, 150e-6, n),
        y=rng.normal(0, 150e-6, n),
        t=rng.normal(0, 1e-12, n),
        px=np.zeros(n),
        py=np.zeros(n),
        pz=np.full(n, 35e6),
        q=np.full(n, charge / n),
        energy=35e6,
    )


# -- the two charge devices ------------------------------------------------

# A Faraday cup is DC coupled; its bandwidth is set by the impedance it is
# discharged through, as the paper notes. 50 ohm into ~100 pF gives ~32 MHz.
faraday_cup = vd.CurrentMonitor(
    rise_time=11e-9,
    droop_time=None,
    transimpedance=50.0,
    noise=2e-4,
    sample_rate=2e10,
    duration=400e-9,
    pretrigger=0.1,
)

# A wall current monitor, bandpass 100 kHz to 6 GHz. The upper corner sets the
# rise time; the lower corner is the droop.
#
# Two things make a WCM behave quite differently from a cup here. Its transfer
# impedance is a couple of ohms, not 50, so only a fraction Zt/50 of the bunch
# charge reaches the integrator -- which is why the paper takes all its WCM data
# on the *highest* sensitivity setting, "due to the very low signal developed by
# the WCM". And the impedance is negative, because a WCM sees the beam's image
# current, which is why the front end's output comes out the other way up
# (Fig. 5) even though nothing in the front end changed.
WCM_LOW, WCM_HIGH = 100e3, 6e9
# Fitted so that 100 pC gives about +400 mV on the highest setting, the
# amplitude reported in Fig. 5. The paper does not state the WCM transfer
# impedance; a few ohms is typical.
WCM_TRANSFER_IMPEDANCE = 2.0  # ohms
wall_current_monitor = vd.CurrentMonitor(
    rise_time=np.log(9.0) / (2 * np.pi * WCM_HIGH),
    droop_time=1.0 / (2 * np.pi * WCM_LOW),
    transimpedance=-WCM_TRANSFER_IMPEDANCE,
    noise=2e-5,
    sample_rate=1e11,
    duration=40e-9,
    pretrigger=0.15,
)


def peak(v):
    """Signed extremum of a waveform."""
    return float(v[np.abs(v).argmax()])


def main() -> None:
    if not vd.ngspice_available():
        raise SystemExit(
            "This example needs ngspice.\n" + vd.spice.INSTALL_HINT
        )
    print(f"ngspice: {vd.ngspice_version()}\n")

    print("charge devices")
    print(
        f"  Faraday cup   DC coupled, {faraday_cup.bandwidth / 1e6:8.2f} MHz, "
        f"rise {faraday_cup.rise_time * 1e9:.1f} ns, 50 ohm"
    )
    print(
        f"  WCM           {WCM_LOW / 1e3:.0f} kHz - {wall_current_monitor.bandwidth / 1e9:.2f} GHz, "
        f"rise {wall_current_monitor.rise_time * 1e12:.1f} ps, "
        f"droop {wall_current_monitor.droop_time * 1e6:.3f} us, "
        f"{WCM_TRANSFER_IMPEDANCE:.0f} ohm"
    )

    # -- 100 pC, each device on the setting the paper used ----------------
    print("\n100 pC, on the setting the paper used for each device")
    traces = {}
    for name, device, setting, reported in (
        ("Faraday cup", faraday_cup, "lowest", -300.0),
        ("WCM", wall_current_monitor, "highest", +400.0),
    ):
        fe = front_end(setting)
        t_in, v_in = device.measure(bunch(100e-12), rng=SEED)
        t_out, v_out = fe.run(t_in, v_in)
        traces[name] = (t_in, v_in, t_out, v_out, setting)
        delivered = np.trapezoid(v_in, t_in) / 50.0
        print(
            f"  {name:12s} {setting:8s} pickup peak {peak(v_in):8.3f} V   "
            f"charge to integrator {delivered * 1e12:7.2f} pC   "
            f"out {peak(v_out) * 1e3:8.2f} mV   (paper: {reported:+.0f} mV)"
        )
    print("  polarity flips for the WCM because it sees the image current")

    # -- linearity across every sensitivity setting, as in Fig. 3 ----------
    print("\ncalibration curves (peak output vs charge), Faraday cup")
    print(f"  {'setting':9s} {'CF':>8s} {'range':>12s} {'V/pC':>9s}  {'linearity':>10s}")
    curves = {}
    for name, (cf, _, _, (lo, hi)) in SETTINGS.items():
        fe = front_end(name)
        charges = np.linspace(max(lo, 1), hi, 7) * 1e-12
        peaks = []
        for q in charges:
            t_in, v_in = faraday_cup.measure(bunch(q), rng=SEED)
            peaks.append(abs(peak(fe.run(t_in, v_in)[1])))
        peaks = np.array(peaks)
        curves[name] = (charges, peaks)
        slope, intercept = np.polyfit(charges * 1e12, peaks, 1)
        residual = peaks - (slope * charges * 1e12 + intercept)
        print(
            f"  {name:9s} {cf * 1e12:7.0f}p {f'{lo}-{hi} pC':>12s} "
            f"{slope * 1e3:8.2f}m {np.abs(residual).max() / peaks.max() * 100:9.3f}%"
        )
    print(f"  each setting spans most of the {FULL_SCALE:.1f} V full scale over its range")

    # -- the 100 kHz corner: baseline droop after the bunch ----------------
    slow = vd.CurrentMonitor(
        rise_time=wall_current_monitor.rise_time,
        droop_time=wall_current_monitor.droop_time,
        transimpedance=-50.0,
        noise=0.0,
        sample_rate=2e9,
        duration=8e-6,
        pretrigger=0.05,
    )
    t_slow, v_slow = slow.measure(bunch(100e-12), rng=SEED)
    bunch_at = int(v_slow.argmin())
    pulse_area = np.trapezoid(v_slow[: bunch_at + 2], t_slow[: bunch_at + 2])
    tail_area = np.trapezoid(v_slow[bunch_at + 2 :], t_slow[bunch_at + 2 :])

    # Fit the tail's decay constant: it should come back as 1/(2 pi f_low).
    tail_t, tail_v = t_slow[bunch_at + 20 :], v_slow[bunch_at + 20 :]
    usable = tail_v > tail_v.max() * 0.05
    fitted_tau = -1.0 / np.polyfit(tail_t[usable], np.log(tail_v[usable]), 1)[0]

    print("\nthe 100 kHz corner, over an 8 us record")
    print(f"  bunch pulse       {v_slow.min() * 1e3:10.3f} mV, area {pulse_area * 1e12:8.3f} pV.s")
    print(f"  droop tail        {v_slow.max() * 1e3:10.3f} mV, area {tail_area * 1e12:8.3f} pV.s")
    print(f"  areas cancel to   {(pulse_area + tail_area) / abs(pulse_area) * 100:9.4f} % "
          f"-- the high pass removes DC exactly")
    print(f"  fitted tail decay {fitted_tau * 1e6:9.4f} us  "
          f"(1/2*pi*{WCM_LOW / 1e3:.0f}kHz = {slow.droop_time * 1e6:.4f} us)")

    # -- figures -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, name in ((axes[0], "Faraday cup"), (axes[1], "WCM")):
        t_in, v_in, t_out, v_out, setting = traces[name]
        vd.plot_signal(t_out, v_out * 1e3, ax=ax, time_units="ns", label="front end out")
        ax.set_ylabel("output (mV)")
        ax.set_title(f"{name}, 100 pC, {setting} sensitivity")
        ax.axhline(0, color="0.7", lw=0.8, zorder=0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/clara_front_end.png", dpi=110)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name in SETTINGS:
        charges, peaks = curves[name]
        ax.plot(charges * 1e12, peaks, "o-", ms=3, label=name)
    ax.axhline(FULL_SCALE, color="0.5", ls="--", lw=0.8, label="full scale")
    ax.set_xlabel("bunch charge (pC)")
    ax.set_ylabel("|peak output| (V)")
    ax.set_title("front end calibration, all sensitivity settings")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/clara_front_end_calibration.png", dpi=110)
    print(f"\nfigures written to {OUT}/")


if __name__ == "__main__":
    main()
