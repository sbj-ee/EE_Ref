# EE-Book

<p align="center">
  <img src="cover.svg" alt="Electrical Engineering Reference — Book Cover" width="400">
</p>

A comprehensive electrical engineering reference covering 19 core disciplines including power systems, communications, semiconductors, control systems, embedded systems, digital logic, circuit analysis, signal processing, electromagnetics, power electronics, instrumentation, electric motors, operational amplifiers, the National Electrical Code, networking, antenna design, radar systems, optics, and engineering economics. Written in Markdown for easy navigation and reference.

## Features

- Power engineering fundamentals including generation, transmission, distribution, protection, power quality, and HVDC
- Communications engineering with analog/digital modulation, channel coding, multiplexing, information theory, noise analysis, and link budgets
- Semiconductor fundamentals (energy bands, doping, PN junctions) and common devices (diodes, transistors, regulators)
- Control systems analysis including PID control, stability analysis, frequency response, and state-space
- Embedded systems including microcontrollers, peripherals, and communication interfaces
- Digital logic from gates and Boolean algebra to sequential circuits and programmable logic
- Circuit analysis fundamentals including DC, AC, and transient analysis
- Signal processing covering Fourier analysis, Laplace/Z-transforms, and digital filters
- Electromagnetics from electrostatics and Maxwell's equations to antennas
- Power electronics including converters, inverters, and thermal management
- Instrumentation and measurement covering sensors, instruments, and data acquisition
- Electric motors including DC, AC, stepper motors, and motor control
- Operational amplifiers covering configurations, active filters, precision rectifiers, oscillators, and real op-amp limitations
- National Electrical Code covering conductor sizing, overcurrent protection, grounding, motors, and transformers
- Networking covering OSI model, fiber optics, DWDM, Ethernet, TCP/UDP, Wi-Fi, and structured cabling
- Antenna design covering wire antennas, aperture antennas, patch antennas, arrays, beamforming, and impedance matching
- Radar systems covering range equation, Doppler, waveforms (pulsed, CW, FMCW), signal processing (MTI, CFAR, SAR), and applications (ATC, weather, automotive, GPR) with comparisons
- Optics covering geometric optics (Snell's law, lenses, mirrors), wave optics (interference, diffraction, polarization), lasers, photodetectors, optical fiber communication, and optical system design
- Engineering economics covering time value of money, NPV/IRR analysis, depreciation, taxes, replacement analysis, inflation, capital budgeting, breakeven/sensitivity analysis, and life cycle cost
- Appendix on imaginary numbers and phasors for AC circuit mathematics
- Appendix on arctangent vs. atan2 for correct angle computation
- Appendix on decibels covering reference levels, arithmetic, and link budgets
- Appendix on unit prefixes and SI units for electrical engineering
- Appendix on getting started with Python and marimo for interactive scripts
- Appendix on matrix operations for circuit analysis and control systems
- Dictionary of 560+ technical terms with definitions and chapter cross-references

## Contents

### Chapter 1: Power Engineering
- Power Generation (fossil fuel, hydro, nuclear, solar, wind, micro grids, GSU transformers, auxiliary transformers)
- Power Transmission (short, medium, long transmission lines, underground transmission)
- Power Distribution (substations, transformers, autotransformers, circuit breakers, voltage regulators, three-phase connections)
- AC Analysis and Power Factor Correction
- Power System Protection (protective relays, fault analysis, protection coordination, symmetrical components)
- Power Quality (harmonics, voltage sags/swells, power quality monitoring, arc flash analysis)
- HVDC Transmission (fundamentals, LCC and VSC converter technologies)
- Load Flow Analysis (power flow equations, bus types, Gauss-Seidel iteration)

### Chapter 2: Communications Engineering
- Analog Modulation (AM Radio, FM Radio)
- Digital Signal Processing (Fourier analysis, Nyquist sampling, quantization)
- Digital Modulation (ASK, FSK, PSK, QAM, constellation diagrams, MIMO and spatial multiplexing)
- Channel Coding and Error Correction (CRC, FEC, interleaving)
- Multiplexing (FDM, OFDM, TDM, CDMA, OFDMA, SC-FDMA)
- Information Theory (Shannon capacity, entropy, source coding)
- Noise in Communication Systems (thermal noise, noise figure, Friis formula, satellite G/T)
- Link Budget Analysis (free-space path loss, receiver sensitivity, fade margin)

### Chapter 3: Semiconductors
- Semiconductor Fundamentals (energy bands, bandgap, doping, drift and diffusion)
- Semiconductor Materials (silicon, gallium arsenide, gallium nitride)
- PN Junction (depletion region, built-in potential, forward/reverse bias, junction capacitance)
- Diodes (rectifier, Zener, Schottky, LEDs)
- Transistors (BJT, MOSFET, power MOSFETs)
- Voltage Regulators (linear regulators, LDO, TI TL431)
- Semiconductor Fabrication (crystal growth, lithography, ion implantation, etching, CMOS integration, FinFET, GAA, chiplets)

### Chapter 4: Control Systems
- Open Loop Systems
- Closed Loop Systems
- Control Signals (step, ramp, impulse, sinusoidal)
- Transfer Functions and Block Diagrams (block diagram algebra, signal flow graphs)
- Time-Domain Performance (first/second-order response, steady-state error)
- PID Control (proportional, integral, derivative, Ziegler-Nichols tuning)
- Stability Analysis (Routh-Hurwitz criterion, root locus)
- Frequency Response Design (Bode plots, gain/phase margins, Nyquist criterion, lead-lag compensation)
- State-Space Representation (state-space model, controllability, observability, state feedback, observer design)
- Digital Control Systems (discrete-time systems, zero-order hold, digital PID implementation)

### Chapter 5: Embedded Systems
- Microcontrollers (architecture, ARM Cortex-M, AVR, PIC, clock system and PLL)
- Memory (Flash, SRAM, EEPROM)
- Peripherals (GPIO, Timers/PWM, ADC, DAC, DMA, watchdog timer)
- Communication Interfaces (UART, SPI, I2C, CAN Bus, USB)
- Interrupts
- Real-Time Operating Systems (RTOS)
- Power Management (sleep modes, wake-up sources, low-power design)
- Development and Debugging (JTAG, SWD, bootloaders)

### Chapter 6: Digital Logic
- Number Systems (binary, hexadecimal, octal)
- Boolean Algebra (truth tables, Karnaugh maps)
- Logic Gates (AND, OR, NOT, NAND, NOR, XOR, logic families and voltage levels)
- Combinational Circuits (adders, multiplexers, decoders, encoders)
- Sequential Circuits (latches, flip-flops, counters, shift registers, state machines)
- Programmable Logic (FPGA, CPLD, timing analysis)

### Chapter 7: Circuit Analysis
- Electric Charge and Current (charge, current, voltage, energy)
- Passive Components (resistors, capacitors, inductors, wire and cable)
- Fundamental Laws (Ohm's Law, KVL, KCL)
- DC Circuit Analysis (series, parallel, series-parallel)
- Analysis Methods (nodal, mesh, superposition, dependent sources)
- Circuit Theorems (Thévenin, Norton, maximum power transfer)
- AC Circuit Analysis (impedance, resonance, AC power, mutual inductance)
- Transient Analysis (RC, RL, RLC circuits)
- Two-Port Networks (Z, Y, h, ABCD parameters, interconnections)

### Chapter 8: Signal Processing
- Signals and Systems (classification, LTI systems, convolution, correlation, sampling theorem and aliasing)
- Fourier Analysis (Fourier series, Fourier transform, DFT, FFT, DCT, Hilbert transform, Goertzel algorithm)
- Laplace Transform (properties, transfer functions, inverse, s-domain circuit analysis)
- Z-Transform (properties, discrete-time transfer functions, inverse Z-transform, bilinear transform, z-plane stability analysis)
- Digital Filters (FIR, IIR, design specifications, multirate signal processing, fixed-point implementation, polyphase structures, allpass filters and group delay equalization)
- Spectral Analysis (PSD, windowing, time-frequency analysis, parametric estimation, cepstral analysis, wavelet transform)
- Adaptive Filtering (LMS algorithm, RLS algorithm, Kalman filter, Wiener filter, compressive sensing, noise cancellation, echo cancellation, channel equalization)

### Chapter 9: Electromagnetics
- Electrostatics (Coulomb's Law, electric field, potential, capacitance, dielectric materials, Laplace's and Poisson's equations)
- Magnetostatics (magnetic fields, Ampère's Law, magnetic force, inductance, magnetic materials and hysteresis, magnetic circuits and core design, eddy currents and induction heating)
- Maxwell's Equations (Gauss's Laws, Faraday's Law, Ampère-Maxwell Law, electromagnetic boundary conditions, electromagnetic potentials)
- Electromagnetic Waves (wave equation, EM spectrum, polarization, skin effect, wave reflection at interfaces, Poynting vector and power flow, wave propagation in lossy media)
- Transmission Lines (characteristic impedance, reflections, Smith Chart, waveguides, microstrip and stripline, transients and TDR, impedance matching, coupled lines and crosstalk)
- Antennas (fundamentals, dipoles, antenna arrays, near-field and far-field regions)
- Electromagnetic Compatibility (shielding effectiveness, EMI filtering and conducted emissions, PCB layout for EMC, ESD protection, grounding and bonding)

### Chapter 10: Power Electronics
- Switching Devices (power diodes, MOSFETs, IGBTs, thyristors, SiC and GaN wide-bandgap semiconductors)
- Rectifiers (single-phase, three-phase, harmonics and input filtering)
- DC-DC Converters (buck, boost, buck-boost, isolated, LLC resonant, voltage/current mode control, multiphase VRMs)
- Inverters (single-phase, three-phase, SPWM, SVPWM, multilevel inverters)
- AC-AC Converters (voltage controllers, cycloconverters, matrix converters)
- Thermal Management (power losses, heat sinks and cooling, TIMs and liquid cooling)
- Power Factor Correction (active PFC, boost PFC, bridgeless topologies, protection circuits)
- Gate Driver Design (bootstrap, isolated drivers, dead-time management)
- EMI and Filtering in Power Converters (conducted/radiated emissions, CISPR, LISN, input filter design)
- Power Supply Protection Circuits (OVP, OCP, soft start, inrush limiting, UVLO)
- Battery Management Systems (cell characteristics, cell balancing, SOC estimation, thermal management, battery protection)
- Battery Energy Storage Systems (BESS architecture, power conversion systems, grid services, sizing and duration, economics and degradation)
- Battery Charging (CC/CV profiles, charger topologies, EV charging levels and standards, fast charging thermal considerations, wireless and inductive charging)
- Supercapacitors (EDLC fundamentals, pseudocapacitors and hybrid supercapacitors, ESR and charge/discharge, series-parallel bank design, regenerative braking and peak power buffering)
- Solar Photovoltaic Systems (PV cell I-V curve and fill factor, maximum power point tracking, grid-tied inverters and anti-islanding, off-grid and hybrid systems, array wiring, shading, and protection)

### Chapter 11: Instrumentation and Measurement
- Measurement Fundamentals (accuracy, precision, resolution, calibration, SNR, uncertainty and GUM, bridges, grounding and shielding)
- Sensors and Transducers (temperature, strain/force, pressure, proximity/position, accelerometers/gyroscopes, magnetic field, current, optical/infrared, flow, chemical and gas sensors)
- Signal Conditioning (amplification, filtering, isolation, linearization, ADC architecture selection)
- Measurement Instruments (DMM, oscilloscope, spectrum analyzer, function generator, power analyzer, LCR meter, frequency counter, VNA, thermal imaging cameras)
- Data Acquisition (sampling/quantization, DAQ systems, data logging and storage, automated test and remote instrument control)

### Chapter 12: Electric Motors
- DC Motors (brushed, brushless BLDC, universal motors)
- AC Motors (induction, synchronous, single-phase, reluctance, linear motors, wound-rotor induction motors)
- Stepper Motors (types, drive modes, applications, resonance and torque curves)
- Motor Control (VFDs, servo systems, soft starters, regenerative braking, FOC, DTC, sensorless control, flux weakening)
- Motor Specifications (nameplate data, motor protection, efficiency standards, selection and sizing, bearings and vibration, thermal modeling, VFD effects on windings)

### Chapter 13: Operational Amplifiers
- Ideal Op-Amp Model (characteristics, open-loop vs. closed-loop)
- Inverting Configurations (amplifier, summing, integrator, differentiator, log/antilog amplifiers, precision rectifier)
- Non-Inverting Configurations (amplifier, voltage follower, transimpedance amplifier, programmable gain amplifier)
- Differential and Instrumentation Amplifiers (difference amplifier, INA, current sense amplifiers)
- Active Filters (first-order, Sallen-Key, state-variable, multiple feedback, notch/band-reject)
- Comparators and Oscillators (basic comparator, Schmitt trigger, Wien bridge, phase-shift oscillators, relaxation oscillator/astable multivibrator)
- Real Op-Amp Limitations (offset/bias, slew rate, CMRR, PSRR, noise analysis, input/output voltage range and rail-to-rail)

### Chapter 14: National Electrical Code (NEC)
- NEC Organization and Structure (chapter hierarchy, key tables)
- Conductor Sizing and Ampacity (Table 310.16, temperature correction, conduit fill, conduit fill calculations)
- Overcurrent Protection (standard ratings, tap rules, coordination, short-circuit calculations, AFCI)
- Grounding and Bonding (system grounding, EGCs, GFPE, bonding, GFCI protection)
- Motor Circuits (FLC tables, conductor sizing, overcurrent, disconnects, VFD circuits)
- Transformer Connections (primary/secondary protection, separately derived systems, delta-wye neutral sizing, K-factor transformers)
- Voltage Drop Calculations (single-phase, three-phase, parallel conductors, conductor sizing for voltage drop, service load calculations)
- Emergency and Standby Power Systems (Articles 700/701/702, generator sizing, transfer switches)

### Chapter 15: Networking
- The OSI Model (layer overview, encapsulation, Layers 1–4)
- Physical Media: Copper (coaxial cable, twisted pair, attenuation, crosstalk, cable testing and certification)
- Physical Media: Fiber Optics (SMF, MMF, transmitters/receivers, link budgets, OTDR troubleshooting)
- Dense Wavelength Division Multiplexing (WDM/DWDM, EDFA, system design)
- Ethernet (frame structure, MAC addressing, physical standards, Power over Ethernet, switching and spanning tree)
- Internet Protocol (IPv4 subnetting, IPv6 addressing, OSPF and BGP routing, NAT/PAT, CIDR and route aggregation)
- Transport Protocols (TCP, UDP, sockets, WSGI, QUIC/HTTP/3)
- Wireless Networking (Wi-Fi standards comparison, RF propagation, wireless security WPA2/WPA3/802.1X)
- Network Infrastructure (switches, routers, patch panels, structured cabling, VLANs and segmentation, network security, SDN)
- Network Performance (latency, throughput, Shannon capacity, BER, timing and synchronization, QoS)

### Chapter 16: Antenna Design and Principles
- Antenna Fundamentals (radiation patterns, antenna parameters, Friis transmission equation)
- Wire Antennas (dipoles, monopoles, loop antennas, Yagi-Uda arrays)
- Aperture Antennas (horn antennas, parabolic reflectors, slot and cavity-backed antennas)
- Printed and Microstrip Antennas (rectangular patch, circular polarization, stacked patches)
- Antenna Arrays (array theory, pattern multiplication, beamforming, massive MIMO, phased array scanning)
- Impedance Matching and Practical Design (VSWR, matching networks, baluns, VNA measurement)

### Chapter 17: Radar Systems
- Radar Fundamentals (range equation, radar cross section, Doppler effect, clutter and noise)
- Radar Waveforms and Modulation (pulsed radar, CW/FMCW, pulse compression)
- Radar Signal Processing (MTI/clutter rejection, CFAR detection, SAR, target tracking)
- Radar Applications (ATC, weather, automotive, GPR, missile defense, LiDAR)
- Radar System Comparison (frequency band selection, cross-application comparison table)

### Chapter 18: Optics
- Geometric Optics (reflection, refraction, Snell's law, lenses, mirrors, prisms)
- Wave Optics (interference, thin-film coatings, multilayer dielectric mirrors, diffraction, polarization)
- Lasers (stimulated emission, laser types, semiconductor lasers, VCSELs, fiber lasers, ultrafast lasers, safety)
- Photodetectors (PIN and avalanche photodiodes, phototransistors, photomultipliers, CCD/CMOS sensors, noise)
- Optical Fiber and Communication (fiber modes, attenuation, dispersion, EDFA, DWDM, coherent detection)
- Optical System Design (radiometry, optical link budgets, lens aberrations, chromatic aberration)

### Chapter 19: Engineering Economics
- Time Value of Money (simple interest, compound interest, nominal/effective rates, continuous compounding)
- Cash Flow Diagrams and Economic Equivalence (conventions, equivalence calculations)
- Single Payment and Uniform Series Factors (F/P, P/F, P/A, A/P, A/F, F/A, arithmetic and geometric gradients)
- Present Worth Analysis (NPV, equal lives, unequal lives with LCM method)
- Annual Worth Analysis (capital recovery, comparing alternatives)
- Rate of Return Analysis (IRR, incremental IRR, MIRR)
- Benefit-Cost Analysis (conventional B/C ratio, incremental B/C)
- Depreciation (straight-line, declining balance, MACRS, units-of-production)
- Taxes and After-Tax Analysis (after-tax cash flow, tax credits, gain/loss on disposal)
- Replacement Analysis (economic service life, defender-challenger, study period)
- Inflation and Price Changes (inflation-adjusted analysis, differential escalation)
- Capital Budgeting and Project Selection (budget constraints, MARR)
- Breakeven and Sensitivity Analysis (breakeven, single-parameter sensitivity, scenario analysis)
- Life Cycle Cost Analysis (LCC components, LCOE)

### Appendix A: Imaginary Numbers and Phasors
- Imaginary Numbers (the imaginary unit, powers of j)
- Complex Numbers (rectangular form, arithmetic, complex conjugate)
- Polar and Exponential Forms (polar form, Euler's formula, conversions)
- Phasors (sinusoidal representation, notation, arithmetic, diagrams)
- Applications in Circuit Analysis (impedance/admittance, voltage/current phasors, power)

### Appendix B: Arctangent and atan2
- The Arctangent Function (definition, range, quadrant ambiguity)
- The Two-Argument Arctangent (full-circle range, quadrant resolution, special cases)
- Applications in Electrical Engineering (impedance angles, phasor angles, power factor)

### Appendix C: Decibels
- Definition and Fundamentals (power ratios, voltage/current ratios, common dB values)
- Absolute Reference Levels (dBm, dBW, dBV, dBμV)
- Decibel Arithmetic (cascaded gains, linear/dB conversion, adding powers)
- Applications (amplifier gain, SNR, link budgets)

### Appendix D: Unit Prefixes and SI Units
- SI Base Units (the seven base units, the ampere and electrical quantities)
- SI Derived Units (voltage/resistance/power, capacitance/inductance/charge, frequency/time)
- SI Prefixes (prefix table, prefix arithmetic, engineering notation)
- Common Unit Conversions (energy/charge units, temperature scales)

### Appendix E: Getting Started with Python and marimo
- Installing Python (download, virtual environments)
- Installing and Running marimo (edit mode, run mode, interface navigation)
- Understanding Script Structure (imports, markdown/code cells, running as scripts)
- Modifying and Extending Scripts (changing parameters, adding cells)

### Appendix F: Matrix Operations
- Matrix Fundamentals (definitions, special matrices)
- Matrix Arithmetic (addition, scalar multiplication, matrix multiplication)
- Matrix Operations (transpose, determinant, inverse)
- Solving Linear Systems (Gaussian elimination, Cramer's rule)
- Applications (nodal analysis, two-port parameters, state-space)

### Appendix G: TCLab Control Lab
- Hardware and Software (TCLab shield, tclab_control package, uv setup)
- Plant Identification (open-loop step test, FOPDT model fit)
- Controller Design (IMC λ-tuning, Ziegler-Nichols comparison)
- Measured Results (single setpoint step response, multi-setpoint schedule)
- Implementation Notes (bumpless transfer, model-hardware discrepancy)

### Dictionary of Terms
- Alphabetical glossary of 560+ technical terms, acronyms, and abbreviations with definitions and chapter references

## Structure

```
EE-Book/
├── 0-Table-of-Contents.md
├── 0-Foreword.md
├── 0-List-of-Figures.md
├── 1-Introduction.md
├── Chapter-1.md   # Power Engineering
├── Chapter-2.md   # Communications Engineering
├── Chapter-3.md   # Semiconductors
├── Chapter-4.md   # Control Systems
├── Chapter-5.md   # Embedded Systems
├── Chapter-6.md   # Digital Logic
├── Chapter-7.md   # Circuit Analysis
├── Chapter-8.md   # Signal Processing
├── Chapter-9.md   # Electromagnetics
├── Chapter-10.md  # Power Electronics
├── Chapter-11.md  # Instrumentation and Measurement
├── Chapter-12.md  # Electric Motors
├── Chapter-13.md  # Operational Amplifiers
├── Chapter-14.md  # National Electrical Code (NEC)
├── Chapter-15.md  # Networking
├── Chapter-16.md  # Antenna Design and Principles
├── Chapter-17.md  # Radar Systems
├── Chapter-18.md  # Optics
├── Chapter-19.md  # Engineering Economics
├── Appendix-A.md  # Imaginary Numbers and Phasors
├── Appendix-B.md  # Arctangent and atan2
├── Appendix-C.md  # Decibels
├── Appendix-D.md  # Unit Prefixes and SI Units
├── Appendix-E.md  # Getting Started with Python and marimo
├── Appendix-F.md  # Matrix Operations
├── Appendix-G.md  # TCLab Control Lab
├── Dictionary.md  # Dictionary of Terms
├── build-pdf.sh   # PDF/LaTeX build script (pandoc + XeLaTeX)
├── pdf-header.tex # LaTeX preamble for Unicode and fonts
├── pdf-sub-sup-filter.lua  # Pandoc Lua filter for sub/sup tags
├── pdf-figure-filter.lua   # Pandoc Lua filter for figure environments
├── pdf-example-box-filter.lua  # Pandoc Lua filter for green example boxes
├── Problems/       # Worked problem sets (173 Markdown files)
├── tests/          # Calculation verification scripts
├── images/         # Generated PNG graphs embedded in chapters
└── scripts/        # Interactive marimo notebooks with graphs
```

## Scripts

The `scripts/` directory contains interactive [marimo](https://marimo.io) notebooks that visualize example problems from the chapters with matplotlib graphs. To run them:

```bash
cd scripts
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
marimo edit 07_circuit_analysis.py
```

| Script | Content |
|--------|---------|
| `01_power_engineering.py` | Generation, transmission, distribution, protection |
| `02_communications.py` | Modulation, channel coding, multiplexing, link budgets |
| `03_semiconductors.py` | Energy bands, PN junctions, diodes, transistors |
| `04_control_systems.py` | PID control, Bode plots, root locus, state-space |
| `05_embedded_systems.py` | Sleep modes, SPI timing, low-power frequency scaling |
| `06_digital_logic.py` | Logic gates, Boolean algebra, sequential circuits |
| `07_circuit_analysis.py` | Impedance, resonance, RC/RL/RLC transients |
| `08_signal_processing.py` | Fourier series/transform, DFT, FIR/IIR filters, PSD |
| `09_electromagnetics.py` | Electrostatics, magnetostatics, transmission lines, EMC |
| `10_power_electronics.py` | Rectifiers, buck/boost converters, SPWM, power losses |
| `11_instrumentation.py` | Sensors, signal conditioning, measurement instruments |
| `12_electric_motors.py` | DC/AC motors, stepper motors, motor control |
| `13_op_amps.py` | Integrator, differentiator, Sallen-Key, Schmitt trigger, slew rate |
| `15_networking.py` | Ethernet frame efficiency, fiber link budget, QoS queuing |
| `16_antenna_design.py` | Dipoles, arrays, patch antennas, beamforming |
| `17_radar_systems.py` | Range equation, Doppler, FMCW, pulse compression |
| `18_optics.py` | Geometric optics, wave optics, lasers, fiber |
| `19_engineering_economics.py` | TVM, NPV, IRR, depreciation, breakeven |
| `14_appendix_a_phasors.py` | Complex plane, phasor addition, phasor diagrams, power triangle |
| `15_appendix_b_atan_vs_atan2.py` | Quadrant ambiguity, atan2 full circle, axis points |
| `16_appendix_c_decibels.py` | dB scale, link budget waterfall, op-amp Bode plot |

## Usage

Download [EE-Book.pdf](EE-Book.pdf) to read the full reference.
