# (c) 2017 Brian Tarasinski
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

import quantumsim.circuit
from quantumsim import sparsedm
import numpy as np
import bitstring

import quantumsim.ptm

import quantumsim.photons

project_z_matrix = np.load('project_z_matrix.npy')

x_bits = ["X%d" % i for i in range(4)]
z_bits = ["Z%d" % i for i in range(4)]
measurement_bits = ["M" + b for b in x_bits + z_bits]
d_bits = ["D%d" % i for i in range(9)]

final_stabilizer_list = [
    np.array(list(bitstring.BitArray(length=5, uint=i)), int)
    for i in range(32)]


def make_new_smearing_matrix(readout_error=None):

    if readout_error is not None:

        def hamming(m, n):
            h, k = 0, m ^ n
            while k:
                h += k & 1
                k >>= 1
            return h
        smearing_matrix = np.zeros((512, 512))
        for i in range(512):
            for j in range(512):
                h = hamming(i, j)
                smearing_matrix[i, j] = readout_error**h * \
                    (1 - readout_error)**(9 - h)
    else:
        smearing_matrix = np.eye(512)

    np.save('smearing_matrix.npy',smearing_matrix)

try:
    smearing_matrix = np.load('smearing_matrix.npy')
except:
    make_new_smearing_matrix(readout_error=0.0015)
    smearing_matrix = np.load('smearing_matrix.npy')

projection_matrix = np.dot(project_z_matrix, smearing_matrix)


def make_circuit(t1=30000, t2=30000, seed=42, t_gate=20,
                 p_x=1e-4, p_yz=5e-4, t_cph_gate=40,
                 static_flux_std=1e-2, t_meas=300, t_cycle=800,
                 readout_error=0, feedback=False):
    surf17 = quantumsim.circuit.Circuit("Surface 17")

    np.random.seed(seed)

    quasi_static_flux = {}

    for b, bm in zip(x_bits + z_bits, measurement_bits):
        surf17.add_qubit(b, t1, t2)
        surf17.add_qubit(bm)
        quasi_static_flux[b] = static_flux_std * np.random.randn()

    for b in d_bits:
        surf17.add_qubit(b, t1, t2)
        quasi_static_flux[b] = static_flux_std * np.random.randn()

    def add_x(c, x_anc, d_bits, anc_pulsed, t):
        t += (t_gate + t_cph_gate) / 2
        for d, apulsed in zip(d_bits, anc_pulsed):
            if d is not None:
                c.add_cphase(x_anc, d, time=t)
                if not apulsed:
                    g = c.add_gate("rotate_z", d, angle=quasi_static_flux[
                                   d], time=t + 0.1)
                    g.label = "x"
            if apulsed:
                g = c.add_gate("rotate_z", x_anc, angle=quasi_static_flux[
                               x_anc], time=t + 0.1)
                g.label = "x"
            t += t_cph_gate

    fg_red = ["D0", "D2", "D6", "D8"]
    fg_red_p = ["D1", "D7"]
    fg_purp = ["D3", "D5"]
    fg_purp_p = ["D4"]
    fg_blue = ["X1", "X3"]
    fg_blue_p = ["X0", "X2"]
    fg_green = ["Z2", "Z3"]
    fg_green_p = ["Z0", "Z1"]

    freq_order = {}
    for b in fg_red + fg_red_p:
        freq_order[b] = 2
    for b in fg_purp + fg_purp_p:
        freq_order[b] = 0
    for b in fg_blue + fg_blue_p + fg_green + fg_green_p:
        freq_order[b] = 1

    # the resonator performs a cphase if the first qubit is pulsed and the
    # second is not.
    resonators = [(anc, dat) if freq_order[anc] > freq_order[dat]
                  else (dat, anc) for anc, dats in
                  [
        ("X0", ["D2", "D1"]),
        ("X1", ["D1", "D0", "D4", "D3"]),
        ("X2", ["D5", "D4", "D8", "D7"]),
        ("X3", ["D7", "D6", ]),
        ("Z0", ["D0", "D3", ]),
        ("Z1", ["D2", "D5", "D1", "D4"]),
        ("Z2", ["D4", "D7", "D3", "D6"]),
        ("Z3", ["D5", "D8"]),

    ] for dat in dats]

    flux_dance_x = [
        fg_red_p + fg_blue_p + fg_purp_p,
        fg_red + fg_blue_p + fg_purp,
        fg_red + fg_blue + fg_purp,
        fg_red_p + fg_blue + fg_purp_p]

    flux_dance_z = [
        fg_red + fg_green + fg_purp,
        fg_red_p + fg_green_p + fg_purp_p,
        fg_red_p + fg_green + fg_purp_p,
        fg_red + fg_green_p + fg_purp,
    ]

    t_next_cphase = (t_gate + t_cph_gate) / 2

    for slice in flux_dance_x:
        for a, d in resonators:
            if a in x_bits or d in x_bits:
                if a in slice and d not in slice:
                    g = quantumsim.circuit.CPhaseRotation(
                        a, d, angle=np.pi + quasi_static_flux[a] / 2, time=t_next_cphase)
                    surf17.add_gate(g)
        for a in slice:
            g = surf17.add_gate("rotate_z", a, angle=quasi_static_flux[a],
                                time=t_next_cphase + 0.1)
            g.label = 'x'
        t_next_cphase += t_cph_gate

    t2 = m_start = t_gate + 4 * t_cph_gate
    t_next_cphase = t2 + (t_gate + t_cph_gate) / 2

    for slice in flux_dance_z:
        for a, d in resonators:
            if a in z_bits or d in z_bits:
                if a in slice and d not in slice:
                    g = quantumsim.circuit.CPhaseRotation(
                        a, d, angle=np.pi + quasi_static_flux[a] / 2, time=t_next_cphase)
                    surf17.add_gate(g)
        for a in slice:
            g = surf17.add_gate("rotate_z", a, angle=quasi_static_flux[a], time=t_next_cphase + 0.1)
            g.label = 'x'
        t_next_cphase += t_cph_gate

    sampler = quantumsim.circuit.BiasedSampler(
        readout_error=readout_error, alpha=1, seed=seed)

    for b in d_bits:
        surf17.add_rotate_y(b, angle=np.pi / 2,
                            dephasing_angle=p_yz, dephasing_axis=p_x, time=0)
        surf17.add_rotate_y(b, angle=-np.pi / 2, dephasing_angle=p_yz,
                            dephasing_axis=p_x, time=4 * t_cph_gate + t_gate)

    for b in x_bits:
        surf17.add_rotate_y(b, angle=np.pi / 2,
                            dephasing_angle=p_yz, dephasing_axis=p_x, time=0)

        normal_rotation =\
            quantumsim.circuit.RotateY(b, angle=-np.pi / 2,
                                       dephasing_angle=p_yz,
                                       dephasing_axis=p_x,
                                       time=4 * t_cph_gate + t_gate)

        cond_gate = normal_rotation

        surf17.add_gate(cond_gate)

    for b in z_bits:
        surf17.add_rotate_y(b, angle=np.pi / 2,
                            dephasing_angle=p_yz, dephasing_axis=p_x, time=t2)
        normal_rotation =\
            quantumsim.circuit.RotateY(b, angle=-np.pi / 2,
                                       dephasing_angle=p_yz,
                                       dephasing_axis=p_x,
                                       time=t2 + 4 * t_cph_gate + t_gate)
        cond_gate = normal_rotation

        surf17.add_gate(cond_gate)

    for b in x_bits:
        m_start = 1.5 * t_gate + 4 * t_cph_gate
        g = quantumsim.circuit.ButterflyGate(
            b, time=m_start, p_exc=0, p_dec=0.005)
        surf17.add_gate(g)
        surf17.add_measurement(b, time=m_start + t_meas, sampler=sampler,
                               output_bit="M"+b)
        g = quantumsim.circuit.ButterflyGate(
            b, time=m_start + 2 * t_meas, p_exc=0, p_dec=0.015)
        surf17.add_gate(g)

    for b in z_bits:
        m_start = t2 + 1.5 * t_gate + 4 * t_cph_gate
        g = quantumsim.circuit.ButterflyGate(
            b, time=m_start, p_exc=0.0, p_dec=0.005)
        surf17.add_gate(g)
        surf17.add_measurement(b, time=m_start + t_meas, sampler=sampler,
                               output_bit="M"+b)
        g = quantumsim.circuit.ButterflyGate(
            b, time=m_start + 2 * t_meas, p_exc=0.0, p_dec=0.015)
        surf17.add_gate(g)

    quantumsim.photons.add_waiting_gates_photons(
        surf17, tmin=0, tmax=t_cycle,
        alpha0=4, kappa=1 / 250, chi=1.3 * 1e-3)

    surf17.order()

    return surf17


def run(seed, cycles=10, padding_to=20, test_data_flag=False,
        error_rate=0.0015, **kwargs):

    circuit = make_circuit(seed=seed, **kwargs)

    sdm = sparsedm.SparseDM(circuit.get_qubit_names())
    for b in ["D%d" % i for i in range(9)]:
        sdm.ensure_dense(b)

    measurements = []
    diagonals = []

    for _ in range(cycles):
        circuit.apply_to(sdm, apply_all_pending=False)

        sdm.renormalize()

        measurement = [sdm.classical[b] for b in measurement_bits]
        measurements.append(measurement)

        diagonal = sdm.full_dm.get_diag()
        diagonals.append(diagonal)

    state = np.random.RandomState(seed=seed + 4*10**8)

    # # # PATCH PART I # # #
    measurements = np.array(measurements)
    new_order = [3, 2, 1, 0, 7, 6, 5, 4]
    measurements = measurements[:,new_order]
    # # # END # # #

    errors = (state.random_sample(len(measurements)*len(measurements[0])) <
              error_rate).tolist()
    measurements = [[measurements[i][j] ^ errors[i*8+j] for j in range(8)]
                    for i in range(len(measurements))]
    
    # # # PATCH PART II # # #
    measurements = np.array(measurements)
    new_order = [3, 2, 1, 0, 7, 6, 5, 4]
    measurements = measurements[:,new_order]
    measurements = [list(mes) for mes in measurements]
    # # # END # # #

    stabilizers = [[m0[j] ^ m1[j] for j in range(8)]
                   for m0, m1 in zip(measurements, [[0]*8] + measurements)]

    events = [[s0[j] ^ s1[j] for j in range(8)]
              for s0, s1 in zip(stabilizers, [[0]*8] + stabilizers)]

    final_stabilizers, parities = get_noisy_measurement(diagonals, seed)
    final_events = [[z1 ^ z2 for z1, z2 in zip(fs, ls[4:])]
                    for fs, ls in zip(final_stabilizers, stabilizers)]

    if test_data_flag is False:
        final_events = final_events[-1]
        final_stabilizers = final_stabilizers[-1]
        parities = parities[-1]

    else:
        events += [[0]*8 for j in range(padding_to-cycles)]

    return measurements, events, final_stabilizers,\
        final_events, parities


def get_noisy_measurement(diagonals, seed):

    state = np.random.RandomState(seed=seed+3*10**8)

    logical_diagonals = [np.dot(projection_matrix, diagonal)
                         for diagonal in diagonals]

    final_stabilizers_and_parities = [final_stabilizer_list[
        state.choice(np.arange(32), p=logical_diagonal)]
        for logical_diagonal in logical_diagonals]

    final_stabilizers = [f[1:] for f in
                         final_stabilizers_and_parities]

    parities = [f[0] for f in final_stabilizers_and_parities]

    return final_stabilizers, parities
