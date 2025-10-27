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

def Databuild(n_samples, backend_name = 'ibm_brisbane',hard_probs=(0.65,0.35)
              ,circuit_probs=(0.5,0.5),prob_depth=(0.2,0.3,0.3,0.2),save_path="../../Dataset"):
    # Create main folder if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

     # Find all folders that match "Sample_n"
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
    
    # Create n_samples new folders and fill them
    for i in range(n_samples):
        sample_num = start_n + i
        sample_folder = os.path.join(save_path, f"Sample_{sample_num}")
        # Create new folder 
        os.makedirs(sample_folder, exist_ok=True)
        backend = Sampling_output_hardware(backend_name,sample_folder,hard_probs)
        qc = Sampling_output_circuit(backend,sample_folder,circuit_probs,prob_depth)
        Output_mapping(backend, qc, sample_folder)
        print(f"Sample {sample_num} created at {sample_folder}")


def Sampling_output_hardware(backend_name, sample_folder, hard_probs=(0.65,0.35)):
    hard_tier = np.random.choice(['Real','Customized'],p=hard_probs)
    print(f"Selected hardware tier: {hard_tier}")
    if hard_tier == 'Real':
        # Load your saved IBM Quantum account
        service = QiskitRuntimeService(channel="ibm_quantum_platform",instance="adevolder")
        backend = service.backend(backend_name)
        Output_real_hardware(backend,sample_folder)
    elif hard_tier == 'Customized':
        backend = CustomizedBackend(backend_name,sample_folder)
    
    return backend
        
    
def CustomizedBackend(backend_name, sample_folder):
    service = QiskitRuntimeService(channel="ibm_quantum_platform",instance="adevolder")
    backend = service.backend(backend_name)
    props = backend.properties()
    n_qubits = len(props.qubits)
    # Extract qubit properties
    T1 = np.zeros(n_qubits)
    T2 = np.zeros(n_qubits)
    freq = np.zeros(n_qubits)
    anha = np.zeros(n_qubits)
    prob_01 = np.zeros(n_qubits)
    prob_10 = np.zeros(n_qubits)
    readout_error = np.zeros(n_qubits)
    for i, qubit_data in enumerate(props.qubits):
        for item in qubit_data:
            if item.name == "T1":
                T1[i] = item.value
            elif item.name == "T2":
                T2[i] = item.value
            elif item.name == "frequency":
                freq[i] = item.value
            elif item.name == "anharmonicity":
                anha[i] = item.value
            elif item.name == "readout_error":
                readout_error[i] = item.value
            elif item.name == "prob_meas0_prep1":
                prob_01[i] = item.value
            elif item.name == "prob_meas1_prep0":
                prob_10[i] = item.value
    
    #Infor for the single qubit error and length
    id_error = np.zeros(n_qubits)
    rz_error = np.zeros(n_qubits)
    sx_error = np.zeros(n_qubits)
    x_error = np.zeros(n_qubits)


    for i, gate_data in enumerate(props.gates):
        if gate_data.gate == "id":
            k = gate_data.qubits[0]
            id_error[k] = gate_data.parameters[0].value
        elif gate_data.gate == "rz":
            k = gate_data.qubits[0]
            rz_error[k] = gate_data.parameters[0].value
        elif gate_data.gate == "sx":
            k = gate_data.qubits[0]
            sx_error[k] = gate_data.parameters[0].value
        elif gate_data.gate == "x":
            k = gate_data.qubits[0]
            x_error[k] = gate_data.parameters[0].value
    
    #Info for two-qubit gates
    ecr_error = []
    for i, gate_data in enumerate(props.gates):
        if gate_data.gate == "ecr":
            ecr_error.append(gate_data.parameters[0].value)

    #Create new backend
    new_backend = copy.deepcopy(backend)

    #Permutation of qubits
    perm = np.random.permutation(n_qubits)
    new_T1 =  T1[perm]
    new_T2 =  T2[perm]
    new_freq =  freq[perm]
    new_anha =  anha[perm]
    new_readout_error =  readout_error[perm]
    new_prob_01 =  prob_01[perm]
    new_prob_10 =  prob_10[perm]
    new_id_error =  id_error[perm]
    new_rz_error =  rz_error[perm]
    new_sx_error =  sx_error[perm]
    new_x_error =  x_error[perm]
    new_ecr_error =  np.random.permutation(ecr_error)
    
    new_props = new_backend.properties()

    #Assign new values to qubits
    for i, qubit_data in enumerate(new_props.qubits):
        for item in qubit_data:
            if item.name == "T1":
                item.value = new_T1[i]
            elif item.name == "T2":
                item.value = new_T2[i]
            elif item.name == "frequency":
                item.value = new_freq[i]
            elif item.name == "anharmonicity":
                item.value = new_anha[i]
            elif item.name == "readout_error":
                item.value = new_readout_error[i]
            elif item.name == "prob_meas0_prep1":
                item.value = new_prob_01[i]
            elif item.name == "prob_meas1_prep0":
                item.value = new_prob_10[i]
    
    #Assign new values to gates
    for i, gate_data in enumerate(new_props.gates):
        if gate_data.gate == "id":
            k = gate_data.qubits[0]
            gate_data.parameters[0].value = new_id_error[k]
        elif gate_data.gate == "rz":
            k = gate_data.qubits[0]
            gate_data.parameters[0].value = new_rz_error[k]
        elif gate_data.gate == "sx":
            k = gate_data.qubits[0]
            gate_data.parameters[0].value = new_sx_error[k]
        elif gate_data.gate == "x":
            k = gate_data.qubits[0]
            gate_data.parameters[0].value = new_x_error[k]
    
    #Assign new values to two-qubit gates
    k=0
    for i, gate_data in enumerate(new_props.gates):
        if gate_data.gate == "ecr":
            gate_data.parameters[0].value = new_ecr_error[k]
            k= k+1
    
    ecr_error_array = np.zeros((n_qubits,n_qubits))
    #Info for gates
    for i, gate_data in enumerate(props.gates):
            if gate_data.gate == "ecr":
                q0, q1 = gate_data.qubits
                ecr_error_array[q0,q1] = gate_data.parameters[0].value
        
    nonzero_indices = np.nonzero(ecr_error_array)
    nonzero_values = ecr_error_array[nonzero_indices]
    # Store as a list of dictionaries
    sparse_ecr = [
        {"row": int(r), "col": int(c), "value": float(v)}
        for r, c, v in zip(nonzero_indices[0], nonzero_indices[1], nonzero_values)
    ]
    new_backend._properties = new_props
    backend = new_backend
    #Getting coupling map and basis gates
    coupling_map = backend.configuration().coupling_map
    basis_gates = backend.configuration().basis_gates

    #Create json file
    hardware_data = {
        "processor_type": f"{backend.configuration().processor_type['family']} {backend.configuration().processor_type['revision']}",
        "n_qubits": n_qubits,
        "basis_gates": basis_gates,
        "coupling_map": coupling_map,
        "T1": new_T1.tolist(),
        "T2": new_T2.tolist(),
        "readout_error": new_readout_error.tolist(),
        "id_error": new_id_error.tolist(),
        "rz_error": new_rz_error.tolist(),
        "sx_error": new_sx_error.tolist(),
        "x_error": new_x_error.tolist(),
        "ecr_error": sparse_ecr,
    }

    # assuming `sample_folder` is something like "../../Dataset/Sample_5"
    file_path = os.path.join(sample_folder, "hardware.json")

    with gzip.open(file_path + ".gz", "wt", encoding="utf-8") as f:
        json.dump(hardware_data, f, indent=2)
    
    return backend

def Output_real_hardware(backend, sample_folder):
    props = backend.properties()
    n_qubits = len(props.qubits)
    # Extract qubit properties
    T1 = np.zeros(n_qubits)
    T2 = np.zeros(n_qubits)
    readout_error = np.zeros(n_qubits)
    for i, qubit_data in enumerate(props.qubits):
        for item in qubit_data:
            if item.name == "T1":
                T1[i] = item.value
            elif item.name == "T2":
                T2[i] = item.value
            elif item.name == "readout_error":
                readout_error[i] = item.value
    
    id_error = np.zeros(n_qubits)
    rz_error = np.zeros(n_qubits)
    sx_error = np.zeros(n_qubits)
    x_error = np.zeros(n_qubits)
    ecr_error = np.zeros((n_qubits,n_qubits))

    #Info for gates
    for i, gate_data in enumerate(props.gates):
        if gate_data.gate == "id":
            k = gate_data.qubits[0]
            id_error[k] = gate_data.parameters[0].value
        elif gate_data.gate == "rz":
            k = gate_data.qubits[0]
            rz_error[k] = gate_data.parameters[0].value
        elif gate_data.gate == "sx":
            k = gate_data.qubits[0]
            sx_error[k] = gate_data.parameters[0].value
        elif gate_data.gate == "x":
            k = gate_data.qubits[0]
            x_error[k] = gate_data.parameters[0].value
        elif gate_data.gate == "ecr":
            q0, q1 = gate_data.qubits
            ecr_error[q0,q1] = gate_data.parameters[0].value
    
    nonzero_indices = np.nonzero(ecr_error)
    nonzero_values = ecr_error[nonzero_indices]
    # Store as a list of dictionaries
    sparse_ecr = [
        {"row": int(r), "col": int(c), "value": float(v)}
        for r, c, v in zip(nonzero_indices[0], nonzero_indices[1], nonzero_values)
    ]
    #Getting coupling map and basis gates
    coupling_map = backend.configuration().coupling_map
    basis_gates = backend.configuration().basis_gates

    #Create json file
    hardware_data = {
        "processor_type": f"{backend.configuration().processor_type['family']} {backend.configuration().processor_type['revision']}",
        "n_qubits": n_qubits,
        "basis_gates": basis_gates,
        "coupling_map": coupling_map,
        "T1": T1.tolist(),
        "T2": T2.tolist(),
        "readout_error": readout_error.tolist(),
        "id_error": id_error.tolist(),
        "rz_error": rz_error.tolist(),
        "sx_error": sx_error.tolist(),
        "x_error": x_error.tolist(),
        "ecr_error": sparse_ecr,
    }
    # assuming `sample_folder` is something like "../../Dataset/Sample_5"
    file_path = os.path.join(sample_folder, "hardware.json")    
    with gzip.open(file_path + ".gz", "wt", encoding="utf-8") as f:
        json.dump(hardware_data, f, indent=2)

def Sampling_output_circuit(backend, sample_folder,circuit_probs,prob_depth):
    # Choice of circuit size
    circuit_tier = np.random.choice(['Famous','Random'],p = circuit_probs)
    print(f"Selected circuit tier: {circuit_tier}") 
    if circuit_tier == 'Famous':
        circuit = Generate_famous_circuit(backend)
    elif circuit_tier == 'Random':
        circuit = Generate_random_circuit(backend,prob_depth)
    
    Output_circuit(circuit,sample_folder,backend)
    return circuit

def Generate_famous_circuit(backend,max_attempts=10):
    # --------------------------------------------------
    # Algorithm list with custom constraints
    # --------------------------------------------------
    ALGORITHMS = {
        "ae": (3, 10),
        "bv": None,
        "dj": None,
        "ghz": None,
        "graphstate": None,
        "grover": (3, 20),
        "hhl": None,
        "qaoa": None,
        "qft": None,
        "qftentangled": None,
        "qnn": None,
        "qpeexact": None,
        "qpeinexact": None,
        "qwalk": None,
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
            print(f"Selected algorithm: {algo}")
            print(f"Number of qubits for the circuit: {n_qubits}")
            qc_perm = random_qubit_permutation(qc)
            return qc_perm

        except Exception as e:
            print(f" ❌ Failed for {algo} ({n_qubits} qubits): {e}")
            continue  # Try another algorithm

    raise RuntimeError(f"Failed to generate a valid circuit after {max_attempts} attempts.")

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
    print(f"Number of qubits to permute: {k}")

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

    return qc_perm

def Generate_random_circuit(backend,prob_depth=(0.2,0.3,0.3,0.2)):
    # Get the number of qubits of the hardware
    n_qubits_hardware = backend.configuration().n_qubits
    # Determine number of qubits
    n_qubits = np.random.randint(3, n_qubits_hardware+1)
    print(f"Number of qubits for the random circuit: {n_qubits}")
    # Determine depth
    tier_depth = np.random.choice(['Shallow','Medium','Deep','Very_deep'], p=prob_depth)
    print(f"Selected depth tier: {tier_depth}")
    if tier_depth == 'Shallow':
        depth = np.random.randint(5, 21)
    elif tier_depth == 'Medium':
        depth = np.random.randint(21, 51)
    elif tier_depth == 'Deep':
        depth = np.random.randint(51, 201)
    elif tier_depth == 'Very_deep':
        depth = np.random.randint(201, 501) 
    
    print(f"Depth of the random circuit: {depth}")
    # Generate random circuit
    qc = random_circuit(n_qubits, depth)
    return qc

def Output_circuit(circuit, sample_folder, backend):
    #Decompose to basis gates
    basis_gates = backend.configuration().basis_gates
    
    # Transpile the circuit to the backend's basis gates
    transpiled_qc = qiskit.transpile(circuit, basis_gates=basis_gates, optimization_level=0)

    n_log_qubits = transpiled_qc.num_qubits

    #Counting the number of X gate for each qubit
    gate = 'x'
    X_counts = np.zeros(n_log_qubits, dtype=int)
    for instr in transpiled_qc.data:
        if len(instr.qubits) == 1 and instr.name in gate:
            qubit_index = transpiled_qc.find_bit(instr.qubits[0]).index
            X_counts[qubit_index] += 1
    
    #Counting the number of SX gate for each qubit
    gate = 'sx'
    SX_counts = np.zeros(n_log_qubits, dtype=int)
    for instr in transpiled_qc.data:
        if len(instr.qubits) == 1 and instr.name in gate:
            qubit_index = transpiled_qc.find_bit(instr.qubits[0]).index
            SX_counts[qubit_index] += 1

    #Counting the number of RZ gate for each qubit
    gate = 'rz'
    RZ_counts = np.zeros(n_log_qubits, dtype=int)
    for instr in transpiled_qc.data:
        if len(instr.qubits) == 1 and instr.name in gate:
            qubit_index = transpiled_qc.find_bit(instr.qubits[0]).index
            RZ_counts[qubit_index] += 1 

    #Counting the number of ECR gate for each pair of qubits
    gate = 'ecr'
    ECR_counts = np.zeros((n_log_qubits, n_log_qubits), dtype=int)
    for instr in transpiled_qc.data:
        if len(instr.qubits) == 2 and instr.name in gate:
            i = transpiled_qc.find_bit(instr.qubits[0]).index
            j = transpiled_qc.find_bit(instr.qubits[1]).index
            ECR_counts[i, j] += 1
    #        two_counts[j, i] += 1  # symmetric, if your 2-qubit gate is undirected
    nonzero_indices = np.nonzero(ECR_counts)
    nonzero_values = ECR_counts[nonzero_indices]
    # Store as a list of dictionaries
    sparse_ecr_circuit = [
        {"row": int(r), "col": int(c), "value": float(v)}
        for r, c, v in zip(nonzero_indices[0], nonzero_indices[1], nonzero_values)
    ]
    #Create json file
    circuit_data = {
        "n_logical_qubits": n_log_qubits,
        "depth": transpiled_qc.depth(),
        "X_counts": X_counts.tolist(),
        "SX_counts": SX_counts.tolist(),
        "RZ_counts": RZ_counts.tolist(),
        "ECR_counts": sparse_ecr_circuit,
    }
    # assuming `sample_folder` is something like "../../Dataset/Sample_5"
    file_path = os.path.join(sample_folder, "circuit.json")
    with gzip.open(file_path + ".gz", "wt", encoding="utf-8") as f:
        json.dump(circuit_data, f, indent=2) 

def Output_mapping(backend, circuit, sample_folder):
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