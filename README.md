# QMapDataset

**QMapDataset** is a Python package to generate datasets for **qubit mapping** and **circuit compilation** studies on quantum hardware. It allows creating datasets for real IBM Quantum backends as well as custom “scrambled” backends, storing circuits, hardware properties, and logical-to-physical qubit mappings in a compressed, JSON-friendly format. This dataset is particularly useful for **machine learning research on qubit mapping and transpilation optimization**.

---

## Features

- Generate **famous quantum circuits** (e.g., Grover, QFT, Shor) or fully **random circuits**.
- Generate hardware backends:  
  - **Real IBM Quantum backends**  
  - **Customized backends** with permuted qubit properties and gate errors
- Automatically store for each sample:  
  - Gate counts (single- and two-qubit)  
  - Circuit depth  
  - Logical → physical qubit mapping after transpilation  
  - Hardware properties including qubit relaxation times, decoherence, and gate errors
- Saves all data in **compressed JSON format** for easy loading

---

## Installation

1. Clone this repository:

```bash
git clone <repo_url>
cd <repo_folder>
```
2. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
A usage example is provided in Generate_dataset.py and in Notebook_datsetbuilding.ipynb.

## Input parameters
- n_samples (int): Number of dataset samples to generate. Each sample will create a folder Sample_n containing hardware, circuit, and mapping JSON files.
- backend_name (str): Name of the IBM Quantum backend to use for circuit transpilation and hardware properties.
- hard_probs (tuple of 2 floats): Probabilities for selecting real vs. customized backend. The default value is (0.65,0.35), i.e a 65% chance for real backend and 35% for customized.
- circuit_probs (tuple of 2 floats): Probabilities for selecting famous vs. random circuit. The default value is (0.5,0.5).
- prob_depth (tuple of 4 floats): Probability distribution for random circuit depth tiers: (Shallow, Medium, Deep, Very_Deep). The default value is (0.25,0.4,0.3,0.05).
- save_path (str): Folder path where the dataset samples will be saved. If the folder does not exist, it will be created.

## Dataset Structure
The code will create folders 'Sample_0', 'Sample_1', ..., each containing:
- hardware.json.gz: qubit properties and gate errors
- circuit.json.gz: gate counts, depth, and algorithm info
- mapping.json.gz: logical-physical qubit mapping after transpilation

### hardware.json.gz
```bash
{
  "tier": "Real" or "Customized",
  "n_qubits": 5,
  "basis_gates": ["u1", "u2", "u3", "cx"],
  "coupling_map": [[0,1], [1,2], ...],
  "T1": [...],
  "T2": [...],
  "single_qubit_errors": {...},
  "multi_qubit_errors": [...]
}
```
### circuit.json.gz
```bash
{
  "circuit_tier": "Famous" or "Random",
  "algorithm_info": [...],
  "n_logical_qubits": 4,
  "depth": 25,
  "single_qubit_counts": {...},
  "two_qubit_counts": [...]
}
```

### mapping.json.gz
```bash
{
  "n_logical_qubits": 4,
  "n_physical_qubits": 5,
  "final_mapping": {"0": 2, "1": 0, "2": 3, "3": 1}
}
```

## Functions Overview
- Databuild(n_samples, ...) – main function to generate multiple dataset samples
- Sampling_output_hardware(...) – choose real or customized backend and generate hardware JSON
- CustomizedBackend(...) – create a backend with permuted qubit properties and save properties of this customized backend
- Output_real_hardware(...) – save properties of a real backend
- Sampling_output_circuit(...) – choose famous or random circuits
- Generate_famous_circuit(...) – generate benchmark circuits from mqt-bench
- random_qubit_permutation(...) – permute a subset of qubits randomly
- Generate_random_circuit(...) – generate random circuits with probabilistic number of qubits and depth
- Output_circuit(...) – transpile circuit, count gates, save circuit info
- Output_mapping(...) – transpile at high optimization to get final qubit mapping

## License
MIT Licence


