import qiskit
from qiskit import QuantumCircuit, Aer, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumRegister, ClassicalRegister
import matplotlib.pyplot as plt
import numpy as np
import json
from itertools import combinations, groupby

# Get the backend to simulate
# backend = Aer.get_backend('statevector_simulator')


def caculate_M_alpha(theta,  alpha, n=1):
    """
    Calculates the theoretical value for stab entropy for a given alpha for example state parameterised by theta.

    :param alpha: which Renyi Entropy ones wants to calculate
    :param theta: angle of the state
    :param n: number of qubits
    :return: M_alpha renyi entropy for p(theta)
    """

    return 0.5 * (1 + (np.cos(theta))**(2 * alpha) + (np.sin(theta)) ** (2 * alpha))


def M2_caculator_qubits(theta, shot_number=1000):
    """
    Runs a simulation to measure the alpha=2 Renyi stabliser entropy for example qubit state parameterised by theta.

    :param theta: angle of the state
    :param shot_number: number of times for the circuit to be repeated
    :return:
    """

    n = 1
    phase = theta

    # ----------------
    # state_generation
    # ----------------

    qrState = QuantumRegister(n)

    qcStateGen = QuantumCircuit(qrState, name="state_generation")

    qcStateGen.h(0)
    qcStateGen.p(phase, 0)

    # qcStateGen.draw('mpl')
    # plt.show()

    # ------------------------------------
    # Load distribution into quantum state
    # -----------------------------------

    # Create control Paulis

    cqX = QuantumCircuit(2)
    cqX.x(0)
    cqX.x(1)

    CX = cqX.to_gate(label="X").control(2, ctrl_state='01')

    cqZ = QuantumCircuit(2)
    cqZ.z(0)
    cqZ.z(1)

    CZ = cqZ.to_gate(label="Z").control(2, ctrl_state='10')

    cqY = QuantumCircuit(2)
    cqY.y(0)
    cqY.y(1)

    CY = cqY.to_gate(label="Y").control(2, ctrl_state='11')

    # Number of ancilla qubits need in circuit
    m = 2*n

    qrLoadControl = QuantumRegister(m, "Control")
    qrLoadOne = QuantumRegister(n)
    qrLoadTwo = QuantumRegister(n)


    qcLoad = QuantumCircuit(qrLoadOne, qrLoadTwo, qrLoadControl)

    qcLoad.h(2)
    qcLoad.h(3)

    qcLoad.append(CX, [2, 3, 1, 0])
    qcLoad.append(CZ, [2, 3, 1, 0])
    qcLoad.append(CY, [2, 3, 1, 0])

    # qcLoad.draw('mpl')
    # plt.show()

    # -------------
    # SWAP TEST
    # -------------

    swapControl = QuantumRegister(1)
    stateOne = QuantumRegister(2)
    stateTwo = QuantumRegister(2)


    qcSwapTest = QuantumCircuit(swapControl, stateOne, stateTwo)

    qcSwapTest.h(0)
    qcSwapTest.cswap(0, 1, 3)
    qcSwapTest.cswap(0, 2, 4)
    qcSwapTest.h(0)

    #  put measurement in master circuit

    # -------------
    # masterCircuit
    # -------------

    qrMaster = QuantumRegister(9)
    crMaster = ClassicalRegister(1)
    qcMaster = QuantumCircuit(qrMaster, crMaster)

    # generate States
    qcMaster = qcMaster.compose(qcStateGen, qubits=[0])
    qcMaster = qcMaster.compose(qcStateGen, qubits=[1])
    qcMaster = qcMaster.compose(qcStateGen, qubits=[2])
    qcMaster = qcMaster.compose(qcStateGen, qubits=[3])

    # Load states
    qcMaster = qcMaster.compose(qcLoad, qubits=[0, 1, 4, 5])
    qcMaster = qcMaster.compose(qcLoad, qubits=[2, 3, 6, 7])

    # SWAP test

    qcMaster = qcMaster.compose(qcSwapTest, qubits=[8, 4, 5, 6, 7])

    qcMaster.measure(8,0)

    # qcMaster.draw('mpl')
    # plt.show()

    # -----------------
    # Simulate Circuit
    # -----------------

    aersim = AerSimulator()
    qcMaster = transpile(qcMaster, aersim)

    # qcMaster.draw('mpl')
    # plt.show()

    # collect results
    result = aersim.run(qcMaster, shots=shot_number, memory=True).result()
    counts = result.get_counts(qcMaster)

    # Caculate Expectation Value

    prob_0 = counts['0'] / shot_number
    prob_1 = counts['1'] / shot_number

    expectation_value = prob_0 * 1 + prob_1 * -1

    final_out = - np.log(2*expectation_value)

    return final_out


def M_alpha_caculator_qubits(theta, alpha, shot_number=1000):
    """
    Runs a simulation to measure the alpha renyi stabliser entropy for example qubit state parameterised by theta.

    :param theta: angle of the state
    :param shot_number: number of times for the circuit to be repeated
    :param alpha: the alpha renyi entropy you want to caculate
    :return: M_alpha
    """

    n = 1 # number of qubits
    phase = theta

    # ----------------
    # state_generation
    # ----------------

    qrState = QuantumRegister(n)

    qcStateGen = QuantumCircuit(qrState, name="state_generation")

    qcStateGen.h(0)
    qcStateGen.p(phase, 0)

    # qcStateGen.draw('mpl')
    # plt.show()

    # ------------------------------------
    # Load distribution into quantum state
    # -----------------------------------

    # Create control Paulis

    # X
    cqX = QuantumCircuit(alpha)
    cqZ = QuantumCircuit(alpha)
    cqY = QuantumCircuit(alpha)

    qubits_control_gate = [el for el in range(0, alpha)]

    cqX.x(qubits_control_gate)
    cqZ.z(qubits_control_gate)
    cqY.y(qubits_control_gate)

    CX = cqX.to_gate(label="X").control(2, ctrl_state='01')
    CZ = cqZ.to_gate(label="Z").control(2, ctrl_state='10')
    CY = cqY.to_gate(label="Y").control(2, ctrl_state='11')

    # Number of ancilla qubits need in circuit
    m = 2 * n

    qrLoadControl = QuantumRegister(m, "Control")
    qrLoadOne = QuantumRegister(alpha)

    qcLoad = QuantumCircuit(qrLoadOne, qrLoadControl)

    qcLoad.h(alpha)
    qcLoad.h(alpha+1)

    qubits_control = [el for el in range(0, alpha)]
    qubits_control.extend([alpha, alpha+1])
    qubits_control.sort(reverse=True)

    qcLoad.append(CX, qubits_control)
    qcLoad.append(CZ, qubits_control)
    qcLoad.append(CY, qubits_control)

    # qcLoad.draw('mpl')
    # plt.show()

    # -------------
    # SWAP TEST
    # -------------

    swapControl = QuantumRegister(1)
    stateOne = QuantumRegister(2)
    stateTwo = QuantumRegister(2)

    qcSwapTest = QuantumCircuit(swapControl, stateOne, stateTwo)

    qcSwapTest.h(0)
    qcSwapTest.cswap(0, 1, 3)
    qcSwapTest.cswap(0, 2, 4)
    qcSwapTest.h(0)


    # -------------
    # masterCircuit
    # -------------

    qrMaster = QuantumRegister(alpha*2 + 5)
    crMaster = ClassicalRegister(1)
    qcMaster = QuantumCircuit(qrMaster, crMaster)

    # generate States
    for el in range(0, alpha*2):
        qcMaster = qcMaster.compose(qcStateGen, qubits=[el])


    # Load states

    first_copies = [el for el in range(0, alpha)]
    second_copies = [el for el in range(alpha, alpha*2)]

    first_copies.extend([alpha*2, alpha*2+1])
    second_copies.extend([alpha*2+2, alpha*2+3])

    qcMaster = qcMaster.compose(qcLoad, qubits=first_copies)
    qcMaster = qcMaster.compose(qcLoad, qubits=second_copies)

    # SWAP test

    qubits_for_swap = [alpha*2+4, alpha*2, alpha*2+1, alpha*2+2, alpha*2+3]

    qcMaster = qcMaster.compose(qcSwapTest, qubits=qubits_for_swap)

    qcMaster.measure(alpha*2+4, 0)


    # -----------------
    # Simulate Circuit
    # -----------------

    aersim = AerSimulator()
    qcMaster = transpile(qcMaster, aersim, seed_transpiler=2000)

    # collect results
    result = aersim.run(qcMaster, shots=shot_number, memory=True).result()
    counts = result.get_counts(qcMaster)

    # Caculate Expectation Value

    prob_0 = counts['0'] / shot_number
    prob_1 = counts['1'] / shot_number

    expectation_value = prob_0 * 1 + prob_1 * -1

    return expectation_value * 2



def add_zero_one(input_list):
    """
    A function to add zero and one to each element of a list of binary strings input

    :param input_list:
    :return:
    """

    bit_list = ['0', '1']

    output_bit_strings = []

    for item in input_list:

        for bit in bit_list:

            new_bit_sting = item + bit

            output_bit_strings.append(new_bit_sting)

    return output_bit_strings


def create_n_bit_strings(n_bits=1):
    """
    A function to create a list of n-bits strings

    :param n_bits: the number of bits in the string
    :return:
    """

    bit_strings = ['0', '1']

    for el in range(0, n_bits-1):

        bit_strings = add_zero_one(bit_strings)

    return bit_strings


def pauliStringCustomGate(num_qubits, gates_list=['I', 'X', 'Y', 'Z']):
    """
    A function to create a custom gate based on an input list of Pauli-gates

    :param num_qubits: the number of target qubits
    :param num_control: the number of control qubits
    :param bit_string: the bit-string to be controlled on
    :param gates_list: the list of Pauli gates to be applied
    :return:
    """

    qc = QuantumCircuit(num_qubits)

    count = 0
    for g in gates_list:
        if g == "I":
            None
        if g == 'X':
            qc.x(count)
        elif g == 'Y':
            qc.y(count)
        elif g == 'Z':
            qc.z(count)
        count += 1

    gate_name = ''
    for el in gates_list:
        gate_name += el

    gate_qc = qc.to_gate(label="{}".format(gate_name))

    return gate_qc, gate_name

def pauliStringCustomGateControl(num_qubits, alpha, num_control, bit_string, gates_list=['I', 'X', 'Y', 'Z']):
    """

    A function to create a custom gate control-gate based on an input list of Pauli-gates that applied to alpha copies

    :param num_qubits:
    :param num_control:
    :param bit_string:
    :return:
    """

    qc_gate_pauli, gate_name = pauliStringCustomGate(num_qubits, gates_list)

    qc2 = QuantumCircuit(num_qubits*alpha)

    for jel in range(0, alpha):
        current_qubits = [el for el in range(jel*num_qubits, (jel+1)*num_qubits)]
        qc2.append(qc_gate_pauli, current_qubits)

    gate_qc = qc2.to_gate(label="{}".format(gate_name)).control(num_control, ctrl_state=bit_string)

    return gate_qc

def add_pauli(input_list):
    """
    A function that adds each pauli to each element of the input list
    :param input_list: a list of paulis
    :return:
    """

    gates_list = [['I'], ['X'], ['Y'], ['Z']]

    output_list = []

    for item in input_list:

        for string in gates_list:

            new_string = item + string
            output_list.append(new_string)

    return output_list


def generate_pauli_group(num_qubits):
    """
    A function to generate a list of the pauli group on n qubits

    :param num_qubits: the number of qubits one wants the pauli group of
    :return:
    """

    pauli_strings = [['I'], ['X'], ['Y'], ['Z']]

    for el in range(0, num_qubits-1):

        pauli_strings = add_pauli(pauli_strings)

    return pauli_strings


def M_alpha_caculator(num_qubits, t_num=1, alpha=1, shot_number=1000):
    """

    Runs a simulation to measure the alpha renyi stabliser entropy via our algorithm for the n-qubit states in
    https://doi.org/10.1038/s41534-022-00666-5.

    :param num_qubits: number of qubits per copy of state
    :param t_num: number of t_gates in the system, must be less that 2num_qubits-1
    :param alpha: the alpha renyi entropy one wants to measure
    :param shot_number: the number of shots to be performed
    :return:
    """

    # ----------------
    # state_generation
    # ----------------

    qrState = QuantumRegister(num_qubits)

    qcStateGen = QuantumCircuit(qrState, name="state_generation")

    list_qubits = [el for el in range(0, num_qubits)]

    # h on all qubits
    qcStateGen.h(list_qubits)

    # T_2 to T_n+2 gates
    for current_t in range(0, t_num-1):

        target_qubit = num_qubits - (current_t+1)

        if target_qubit >= 0:

            qcStateGen.t(target_qubit)

    # add first row of cx gates
    list_qubits_cx = [(el, el+1) for el in range(0, num_qubits-1)]

    for qubits in list_qubits_cx:

        qcStateGen.cx(qubits[0], qubits[1])

    # T_1 gate
    qcStateGen.t(num_qubits - 1)

    # remaining T gates
    remaining_t = t_num-num_qubits-1


    if remaining_t > 0:

        for current_t in range(0, remaining_t):

            target_qubit = num_qubits - current_t - 2

            if target_qubit >= 0:
                qcStateGen.t(target_qubit)

    # add second row of cx gates
    list_qubits_cx.sort(reverse=True)

    for qubits in list_qubits_cx:

        qcStateGen.cx(qubits[0], qubits[1])

    # qcStateGen.draw('mpl')
    # plt.show()

    # ------------------------------------
    # Load distribution into quantum state
    # -----------------------------------

    # Number of ancilla qubits need in circuit
    num_ancilla = 2 * num_qubits

    # Create control Paulis

    # list of paulis in n qubit group
    pauli_group_list = generate_pauli_group(num_qubits)

    # m length bit strings list
    bit_strings = create_n_bit_strings(num_ancilla)

    # create Pauli control gates

    pauli_control_gates = [pauliStringCustomGateControl(num_qubits, alpha, num_ancilla, el, jel) for el, jel in zip(bit_strings, pauli_group_list)]

    # create load
    qrLoadControl = QuantumRegister(num_ancilla, "Control")
    qrLoadtarget = QuantumRegister(num_qubits*alpha)  # one register per copy

    qcLoad = QuantumCircuit(qrLoadtarget, qrLoadControl)

    # H on all controls

    control_qubits = [el for el in range(num_qubits*alpha, num_qubits*alpha + num_ancilla)]

    # print(target_qubits)
    qcLoad.h(control_qubits)

    # qcLoad.draw('mpl')
    # plt.show()

    target_qubits = [el for el in range(alpha*num_qubits)]
    for control_gate in pauli_control_gates[1:]:
        # print(control_gate)
        qcLoad.append(control_gate, control_qubits+target_qubits)

    # qcLoad.draw('mpl')
    # plt.show()

    # -------------
    # SWAP TEST
    # -------------

    swapControl = QuantumRegister(1)
    stateOne = QuantumRegister(num_ancilla)
    stateTwo = QuantumRegister(num_ancilla)


    qcSwapTest = QuantumCircuit(swapControl, stateOne, stateTwo)

    qcSwapTest.h(0)

    for el in range(1, num_ancilla+1):
        qcSwapTest.cswap(0, el, num_ancilla+el)

    qcSwapTest.h(0)

    #  put measurement in master circuit

    # -------------
    # masterCircuit
    # -------------

    two_alpha = alpha * 2

    qrMaster = QuantumRegister(num_qubits*two_alpha + num_ancilla*2 + 1)
    crMaster = ClassicalRegister(1)
    qcMaster = QuantumCircuit(qrMaster, crMaster)

    # generate States

    for jel in range(0, two_alpha):
        current_qubits = [el for el in range(jel*num_qubits, (jel+1)*num_qubits)]

        qcMaster = qcMaster.compose(qcStateGen, qubits=current_qubits)

    # Load states

    first_anicallas = [el for el in range(num_qubits*two_alpha, num_qubits*two_alpha + num_ancilla)]
    second_anicallas = [el for el in range(num_qubits*two_alpha + num_ancilla, num_qubits*two_alpha + num_ancilla*2)]

    qubits_copies_one = [el for el in range(0, num_qubits*alpha)]
    qubits_copies_two = [el for el in range(num_qubits*alpha, num_qubits*two_alpha)]

    first = qubits_copies_one + first_anicallas
    second = qubits_copies_two + second_anicallas

    qcMaster = qcMaster.compose(qcLoad, qubits=first)
    qcMaster = qcMaster.compose(qcLoad, qubits=second)

    # SWAP test

    ancillas_for_swap = first_anicallas + second_anicallas

    final_qubit = num_qubits*two_alpha + num_ancilla*2

    swap_qubits = [final_qubit] + ancillas_for_swap

    qcMaster = qcMaster.compose(qcSwapTest, qubits=swap_qubits)

    qcMaster.measure(final_qubit, 0)

    # qcMaster.draw('mpl')
    # plt.show()

    # -----------------
    # Simulate Circuit
    # -----------------
    #
    # qcMaster.draw('mpl')
    # plt.show()

    aersim = AerSimulator()
    qcMaster = transpile(qcMaster, aersim, seed_transpiler=2000)

    # qcMaster.draw('mpl')
    # plt.show()

    # collect results
    result = aersim.run(qcMaster, shots=shot_number, memory=True).result()
    counts = result.get_counts(qcMaster)

    # Calculate Expectation Value

    prob_0 = counts['0'] / shot_number
    prob_1 = counts['1'] / shot_number

    expectation_value = prob_0 * 1 + prob_1 * -1

    final_out = (1 / (1 - alpha)) * np.log(2**num_qubits * expectation_value)

    return final_out


data_points = 60
shots_base = 16000
error = 0.05

thetas = np.linspace(0, np.pi, data_points)
eplisionError = [error/2 for el in thetas]


# Run Simulations

methods_2 = [M_alpha_caculator_qubits(el, 2, shot_number=shots_base*2) for el in thetas]
methods_3 = [M_alpha_caculator_qubits(el, 3, shot_number=shots_base*3) for el in thetas]
methods_5 = [M_alpha_caculator_qubits(el, 5, shot_number=shots_base*5) for el in thetas]
methods_7 = [M_alpha_caculator_qubits(el, 7, shot_number=shots_base*7) for el in thetas]


# Collect Theory Data 

methods_theory_2 = [caculate_M_alpha(el, 2) for el in thetas]
methods_theory_3 = [caculate_M_alpha(el, 3) for el in thetas]
methods_theory_5 = [caculate_M_alpha(el, 5) for el in thetas]
methods_theory_7 = [caculate_M_alpha(el, 7) for el in thetas]

# Save Data

# with open("./data/methods_theory_2.json", 'w') as f:

#     json.dump(methods_theory_2, f, indent=2)

# with open("./data/methods_theory_3.json", 'w') as f:

#     json.dump(methods_theory_3, f, indent=2)

# with open("./data/methods_theory_5.json", 'w') as f:

#     json.dump(methods_theory_5, f, indent=2)

# with open("./data/methods_theory_7.json", 'w') as f:

#     json.dump(methods_theory_7, f, indent=2)


