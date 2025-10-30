import numpy as np
import qiskit
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.circuit.random import random_circuit
import os
import re
import copy
import json
import gzip
from mqt.bench import get_benchmark, BenchmarkLevel
from collections import defaultdict
# =========================================================
# Main function to generate a dataset of samples
# =========================================================

def Databuild(n_samples, backend_name = 'ibm_brisbane',hard_probs=(0.65,0.35)
              ,circuit_probs=(0.5,0.5),prob_depth=(0.25,0.4,0.3,0.05),save_path="../../Dataset"):
    """
    Generate n_samples of qubit mapping datasets. Each sample contains:
      - A hardware configuration (real or customized)
      - A quantum circuit (famous or random)
      - Mapping from logical to physical qubits
    
    Parameters:
        n_samples: number of new samples to generate
        backend_name: IBM backend name
        hard_probs: probabilities for choosing 'Real' or 'Customized' backend
        circuit_probs: probabilities for choosing 'Famous' or 'Random' circuit
        prob_depth: probability distribution for circuit depth tiers
        save_path: folder where dataset is stored
    """
    # Create main folder if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Find all existing sample folders to continue numbering
    pattern = re.compile(r"Sample_(\d+)")
    samples = []

    for name in os.listdir(save_path):
        full_path = os.path.join(save_path, name)
        if os.path.isdir(full_path):   # ✅ Use full path here
            match = pattern.match(name)
            if match:
                samples.append(int(match.group(1)))
    
    
    # Determine starting index
    start_n = max(samples) + 1 if samples else 0
    
    # Generate new samples
    for i in range(n_samples):
        sample_num = start_n + i
        sample_folder = os.path.join(save_path, f"Sample_{sample_num}")
        # Create new folder 
        os.makedirs(sample_folder, exist_ok=True)
        # Sample hardware configuration
        backend = Sampling_output_hardware(backend_name,sample_folder,hard_probs)
        # Sample quantum circuit
        qc = Sampling_output_circuit(backend,sample_folder,circuit_probs,prob_depth)
        # Compute mapping from logical to physical qubits
        Output_mapping(backend, qc, sample_folder)
        print(f"Sample {sample_num} created at {sample_folder}")


def Sampling_output_hardware(backend_name, sample_folder, hard_probs=(0.65,0.35)):
    """
    Randomly select between real IBM backend or customized backend
    """
    hard_tier = np.random.choice(['Real','Customized'],p=hard_probs)
    if hard_tier == 'Real':
        # Load your saved IBM Quantum account
        service = QiskitRuntimeService(channel="ibm_quantum_platform",instance="adevolder")
        backend = service.backend(backend_name)
        Output_real_hardware(backend,sample_folder,hard_tier)
    elif hard_tier == 'Customized':
        backend = CustomizedBackend(backend_name,sample_folder,hard_tier)
    
    return backend
        
# =========================================================
# Customize backend by permuting qubits and gate errors
# =========================================================
    
def CustomizedBackend(backend_name, sample_folder,hard_tier):
    service = QiskitRuntimeService(channel="ibm_quantum_platform",instance="adevolder")
    backend = service.backend(backend_name)
    props = backend.properties()
    n_qubits = len(props.qubits)
    # --------------------------------------------------
    # Extract qubit properties dynamically
    # --------------------------------------------------
    qubit_props = {name: np.zeros(n_qubits) for name in ["T1", "T2", "frequency", "anharmonicity", "readout_error","prob_meas0_prep1", "prob_meas1_prep0"]}

    for i, qubit_data in enumerate(props.qubits):
        for item in qubit_data:
            if item.name in qubit_props:
                qubit_props[item.name][i] = item.value
    
    # --------------------------------------------------
    # Extract gate errors dynamically
    # --------------------------------------------------
    single_qubit_errors = defaultdict(lambda: np.zeros(n_qubits))
    multi_qubit_errors = defaultdict(list)  # store values for multi-qubit gates like 'ecr'

    for gate_data in props.gates:
        qubits = gate_data.qubits
        gate_name = gate_data.gate
        value = gate_data.parameters[0].value  # assuming first parameter is the error
        if len(qubits) == 1:
            single_qubit_errors[gate_name][qubits[0]] = value
        else:
            multi_qubit_errors[gate_name].append(value)

    # --------------------------------------------------
    # Permute qubits randomly
    # --------------------------------------------------
    new_backend = copy.deepcopy(backend)
    perm = np.random.permutation(n_qubits)

    # Permute qubit properties
    for key in qubit_props:
        qubit_props[key] = qubit_props[key][perm]

    # Permute single-qubit errors
    for key in single_qubit_errors:
        single_qubit_errors[key] = single_qubit_errors[key][perm]

    # Permute multi-qubit errors (shuffle list)
    for key in multi_qubit_errors:
        multi_qubit_errors[key] = np.random.permutation(multi_qubit_errors[key])
    
    # --------------------------------------------------
    # Assign permuted properties back to backend
    # --------------------------------------------------
    new_props = new_backend.properties()

    # Qubit properties
    for i, qubit_data in enumerate(new_props.qubits):
        for item in qubit_data:
            if item.name in qubit_props:
                item.value = qubit_props[item.name][i]

    # Gate properties
    multi_gate_counters = {key: 0 for key in multi_qubit_errors}  # track index when assigning
    for gate_data in new_props.gates:
        qubits = gate_data.qubits
        gate_name = gate_data.gate
        if len(qubits) == 1:
            gate_data.parameters[0].value = single_qubit_errors[gate_name][qubits[0]]
        else:
            idx = multi_gate_counters[gate_name]
            gate_data.parameters[0].value = multi_qubit_errors[gate_name][idx]
            multi_gate_counters[gate_name] += 1

    new_backend._properties = new_props
    
    # --------------------------------------------------
    # Prepare JSON-friendly sparse representation for multi-qubit gates
    # --------------------------------------------------
    sparse_multi_qubit = {}
    for gate_name, values in multi_qubit_errors.items():
        error_array = np.zeros((n_qubits, n_qubits))
        idx = 0
        for gate_data in props.gates:
            if gate_data.gate == gate_name and len(gate_data.qubits) > 1:
                q0, q1 = gate_data.qubits
                error_array[q0, q1] = gate_data.parameters[0].value
                idx += 1
        rows, cols = np.nonzero(error_array)
        sparse_multi_qubit[gate_name] = [
            {"row": int(r), "col": int(c), "value": float(v)}
            for r, c, v in zip(rows, cols, error_array[rows, cols])
        ]

    # --------------------------------------------------
    # Create hardware JSON
    # --------------------------------------------------
    hardware_data = {
        "tier": hard_tier,
        "processor_type": f"{backend.configuration().processor_type['family']} {backend.configuration().processor_type['revision']}",
        "n_qubits": n_qubits,
        "basis_gates": backend.configuration().basis_gates,
        "coupling_map": backend.configuration().coupling_map,
    }

    # Add qubit properties
    hardware_data.update({k: v.tolist() for k, v in qubit_props.items()})

    # Add single-qubit errors
    for k, v in single_qubit_errors.items():
        hardware_data[f"{k}_error"] = v.tolist()

    # Add multi-qubit sparse errors
    hardware_data.update(sparse_multi_qubit)

    # --------------------------------------------------
    # Save JSON compressed
    # --------------------------------------------------
    file_path = os.path.join(sample_folder, "hardware.json")
    with gzip.open(file_path + ".gz", "wt", encoding="utf-8") as f:
        json.dump(hardware_data, f, indent=2)

    return new_backend

# =========================================================
# Save real IBM backend hardware properties
# =========================================================

def Output_real_hardware(backend, sample_folder, hard_tier):
    """
    Extract qubit and gate properties from a real IBM backend
    and save as JSON
    """
    props = backend.properties()
    n_qubits = len(props.qubits)

    # --------------------------------------------------
    # Extract qubit properties dynamically
    # --------------------------------------------------
    qubit_props = {name: np.zeros(n_qubits) for name in ["T1", "T2", "readout_error",
                                                        "prob_meas0_prep1", "prob_meas1_prep0",
                                                        "frequency", "anharmonicity"]}

    for i, qubit_data in enumerate(props.qubits):
        for item in qubit_data:
            if item.name in qubit_props:
                qubit_props[item.name][i] = item.value

    # --------------------------------------------------
    # Extract gate errors dynamically
    # --------------------------------------------------
    single_qubit_errors = defaultdict(lambda: np.zeros(n_qubits))
    multi_qubit_errors = defaultdict(lambda: np.zeros((n_qubits, n_qubits)))

    for gate_data in props.gates:
        qubits = gate_data.qubits
        gate_name = gate_data.gate
        value = gate_data.parameters[0].value  # first parameter is usually the error
        if len(qubits) == 1:
            single_qubit_errors[gate_name][qubits[0]] = value
        else:
            q0, q1 = qubits
            multi_qubit_errors[gate_name][q0, q1] = value

    # --------------------------------------------------
    # Convert multi-qubit errors to sparse format
    # --------------------------------------------------
    sparse_multi_qubit = {}
    for gate_name, matrix in multi_qubit_errors.items():
        rows, cols = np.nonzero(matrix)
        values = matrix[rows, cols]
        sparse_multi_qubit[gate_name] = [
            {"row": int(r), "col": int(c), "value": float(v)}
            for r, c, v in zip(rows, cols, values)
        ]

    # --------------------------------------------------
    # Build final hardware dictionary
    # --------------------------------------------------
    hardware_data = {
        "tier": hard_tier,
        "processor_type": f"{backend.configuration().processor_type['family']} {backend.configuration().processor_type['revision']}",
        "n_qubits": n_qubits,
        "basis_gates": backend.configuration().basis_gates,
        "coupling_map": backend.configuration().coupling_map,
    }

    # Add qubit properties
    hardware_data.update({k: v.tolist() for k, v in qubit_props.items()})

    # Add single-qubit errors
    for k, v in single_qubit_errors.items():
        hardware_data[f"{k}_error"] = v.tolist()

    # Add multi-qubit sparse errors
    hardware_data.update(sparse_multi_qubit)

    # --------------------------------------------------
    # Save JSON compressed
    # --------------------------------------------------
    file_path = os.path.join(sample_folder, "hardware.json")
    with gzip.open(file_path + ".gz", "wt", encoding="utf-8") as f:
        json.dump(hardware_data, f, indent=2)

def Sampling_output_circuit(backend, sample_folder,circuit_probs,prob_depth):
    """
    Generate either a famous or random quantum circuit
    """
    # Choice of circuit size
    circuit_tier = np.random.choice(['Famous','Random'],p = circuit_probs)
#    print(f"Selected circuit tier: {circuit_tier}") 
    if circuit_tier == 'Famous':
        circuit, algo_info = Generate_famous_circuit(backend)
        if circuit is None:
            print("Famous circuit generation failed, generating a Random circuit instead.")
            circuit, algo_info = Generate_random_circuit(backend,prob_depth)
    elif circuit_tier == 'Random':
        circuit, algo_info = Generate_random_circuit(backend,prob_depth)
    
    Output_circuit(circuit,sample_folder,backend,circuit_tier,algo_info)
    return circuit

def Generate_famous_circuit(backend,max_attempts=3):
    """
    Generate a "famous" quantum circuit from MQT benchmarks.
    Tries up to `max_attempts` to create a valid circuit compatible with the backend.
    Returns a permuted version of the circuit and information about the algorithm.
    """
    # --------------------------------------------------
    # Predefined algorithms and optional qubit constraints
    # None = no constraint on number of qubits
    # Tuple = (min_qubits, max_qubits)
    ALGORITHMS = {
        "ae": (3, 10),
        "bv": None,
        "dj": None,
        "ghz": None,
        "graphstate": None,
        "grover": (3, 10),
        "hhl": None,
        "qaoa": None,
        "qft": None,
        "qftentangled": None,
        "qnn": None,
        "qpeexact": None,
        "qpeinexact": None,
        "shor": (18, 18),
        "vqe_real_amp": None,
        "vqe_su2": None,
        "vqe_two_local": None,
        "wstate": None,
    }
    # Get the number of qubits of the hardware
    n_qubits_hardware = backend.configuration().n_qubits
    
    for attempt in range(max_attempts):
        # Select a random algorithm
        algo = np.random.choice(list(ALGORITHMS.keys())) 
        # Get constraints
        constraints = ALGORITHMS[algo]
        # Determine number of qubits
        if constraints is None:
            n_qubits = np.random.randint(3, n_qubits_hardware)
        else:
            min_qubits, max_qubits = constraints
            max_qubits = min(max_qubits, n_qubits_hardware)
            n_qubits = np.random.randint(min_qubits, max_qubits + 1)
    
    
        try:
            qc = get_benchmark(algo, circuit_size=n_qubits, level=BenchmarkLevel.INDEP)
            algo_info = []
            algo_info.append(f"Algorithm: {algo}, Qubits: {n_qubits}")
#            print(f"Selected algorithm: {algo}")
#            print(f"Number of qubits for the circuit: {n_qubits}")
            qc_perm, n_perm, perm = random_qubit_permutation(qc)
            algo_info.append(f"Number of Qubit permutation: {n_perm} qubits permuted, Permutation: {perm}")
            return qc_perm, algo_info

        except Exception as e:
            continue  # Try another algorithm

    print(f"⚠️ Could not generate a valid famous circuit after {max_attempts} attempts.")
    return None

def random_qubit_permutation(qc, seed=None):
    """
    Returns a new circuit equivalent to `qc` but with a random permutation
    applied to a random subset of its qubits.
    - The number of permuted qubits k is randomly chosen between 2 and n_qubits.
    - Only those k qubits are permuted, others remain in place.
    """
    rng = np.random.default_rng(seed)
    n_qubits = qc.num_qubits

    # Step 1 — choose how many qubits will be permuted (at least 2)
    k = rng.integers(2, n_qubits + 1) if n_qubits >= 2 else 1
#    print(f"Number of qubits to permute: {k}")

    # Step 2 — choose which qubits are permuted
    permuted_indices = rng.choice(n_qubits, size=k, replace=False)
    
    # Step 3 — create a random permutation for those k qubits
    shuffled = rng.permutation(permuted_indices)

    # Step 4 — build full permutation (identity except for permuted subset)
    perm = list(range(n_qubits))
    for i, j in zip(permuted_indices, shuffled):
        perm[i] = j

    # Create a mapping from old to new qubits
    mapping = {old: new for old, new in enumerate(perm)}

    # Create a new empty circuit with same structure
    qc_perm = qiskit.QuantumCircuit(n_qubits, qc.num_clbits)

    # Apply the same operations but remap qubits
    for instruction in qc.data:
        inst = instruction.operation
        qargs = instruction.qubits
        cargs = instruction.clbits

        new_qargs = [qc_perm.qubits[mapping[qc.find_bit(q).index]] for q in qargs]
        new_cargs = [qc_perm.clbits[qc.find_bit(c).index] for c in cargs]
        qc_perm.append(inst, new_qargs, new_cargs)

    return qc_perm, k, perm

def Generate_random_circuit(backend,prob_depth=(0.2,0.3,0.3,0.2)):
    """
    Generate a random quantum circuit compatible with the backend.
    Randomly chooses:
      - Number of qubits
      - Circuit depth tier (Shallow, Medium, Deep, Very Deep)
    Returns the circuit and metadata info.
    """
    # Get the number of qubits of the hardware
    n_qubits_hardware = backend.configuration().n_qubits
    # Determine number of qubits
    n_qubits = np.random.randint(3, n_qubits_hardware+1)
    algo_info = []
    algo_info.append(f"Qubits: {n_qubits}")
 #   print(f"Number of qubits for the random circuit: {n_qubits}")
    # Determine depth
    tier_depth = np.random.choice(['Shallow','Medium','Deep','Very_deep'], p=prob_depth)
#    print(f"Selected depth tier: {tier_depth}")
    if tier_depth == 'Shallow':
        depth = np.random.randint(5, 16)
    elif tier_depth == 'Medium':
        depth = np.random.randint(16, 51)
    elif tier_depth == 'Deep':
        depth = np.random.randint(51, 151)
    elif tier_depth == 'Very_deep':
        depth = np.random.randint(151, 401) 
    
#    print(f"Depth of the random circuit: {depth}")
    algo_info.append(f"Depth: {depth}")
    # Generate random circuit
    qc = random_circuit(n_qubits, depth)
    return qc, algo_info

def Output_circuit(circuit, sample_folder, backend,circuit_tier,algo_info):
    """
    Transpile the circuit to the backend's basis gates and
    count all gate occurrences (single and two-qubit).
    Saves circuit info and gate counts as compressed JSON.
    """
    #Decompose to basis gates
    basis_gates = backend.configuration().basis_gates
    
    # Transpile the circuit to the backend's basis gates
    transpiled_qc = qiskit.transpile(circuit, basis_gates=basis_gates, optimization_level=0)

    n_log_qubits = transpiled_qc.num_qubits

    # --------------------------------------------------
    # Initialize dynamic counters
    # --------------------------------------------------
    # Single-qubit gates: dictionary of arrays
    single_qubit_counters = defaultdict(lambda: np.zeros(n_log_qubits, dtype=int))
    # Two-qubit gates: dictionary of matrices
    two_qubit_counters = defaultdict(lambda: np.zeros((n_log_qubits, n_log_qubits), dtype=int))


    # --------------------------------------------------
    # Count gates dynamically
    # --------------------------------------------------
    for instr in transpiled_qc.data:
        name = instr.name
        qubits = [transpiled_qc.find_bit(q).index for q in instr.qubits]
        
        if len(qubits) == 1:
            single_qubit_counters[name][qubits[0]] += 1
        elif len(qubits) == 2:
            i, j = qubits
            two_qubit_counters[name][i, j] += 1

    # --------------------------------------------------
    # Convert two-qubit counters to sparse JSON-friendly format
    # --------------------------------------------------
    sparse_two_qubit = {}
    for gate, matrix in two_qubit_counters.items():
        rows, cols = np.nonzero(matrix)
        values = matrix[rows, cols]
        sparse_two_qubit[gate] = [
            {"row": int(r), "col": int(c), "value": float(v)}
            for r, c, v in zip(rows, cols, values)
        ]

    # --------------------------------------------------
    # Prepare final dictionary
    # --------------------------------------------------
    circuit_data = {
        "circuit_tier": circuit_tier,
        "algorithm_info": algo_info,
        "n_logical_qubits": n_log_qubits,
        "depth": transpiled_qc.depth(),
        "single_qubit_counts": {gate: counts.tolist() for gate, counts in single_qubit_counters.items()},
        "two_qubit_counts": sparse_two_qubit,
    }

    # --------------------------------------------------
    # Save as compressed JSON
    # --------------------------------------------------
    file_path = os.path.join(sample_folder, "circuit.json")
    with gzip.open(file_path + ".gz", "wt", encoding="utf-8") as f:
        json.dump(circuit_data, f, indent=2)

def Output_mapping(backend, circuit, sample_folder):
    """
    Transpile circuit to backend to get optimized mapping of logical -> physical qubits.
    Saves mapping info as compressed JSON.
    """
    n_log_qubits = circuit.num_qubits
    n_phys_qubits = backend.configuration().n_qubits
    #transpile with highest optimization level to get mapping
    transpiled_qc = qiskit.transpile(circuit, backend=backend, optimization_level=3)
    #Extract mapping
    layout = transpiled_qc.layout
    Final_layout=layout.final_index_layout()
    #Create json file
    mapping_data = {
        "n_logical_qubits": n_log_qubits,
        "n_physical_qubits": n_phys_qubits,
        "final_mapping": Final_layout,
    }   
    # assuming `sample_folder` is something like "../../Dataset/Sample_5"
    file_path = os.path.join(sample_folder, "mapping.json")

    with gzip.open(file_path + ".gz", "wt", encoding="utf-8") as f:
        json.dump(mapping_data, f, indent=2)