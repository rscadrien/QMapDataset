import os
from Dataset_build import Databuild
# ---------------------------------
# 1. Define dataset parameters
# ---------------------------------
n_samples = 100                # number of samples to generate
backend_name = "ibm_brisbane" # IBM backend to use
hard_probs = (0.65, 0.35)       # probability of Real vs Customized backend
circuit_probs = (0.5, 0.5)    # probability of Famous vs Random circuits
prob_depth = (0.25, 0.4, 0.3, 0.05)  # depth tiers for Random circuits
save_path = "../../Dataset/IBM_Brisbane"       # folder where samples will be saved
# ---------------------------------
# 2. Generate the dataset
# ---------------------------------
Databuild(
    n_samples=n_samples,
    backend_name=backend_name,
    hard_probs=hard_probs,
    circuit_probs=circuit_probs,
    prob_depth=prob_depth,
    save_path=save_path
)