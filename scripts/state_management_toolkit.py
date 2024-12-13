import pandas as pd
from hmm_refinement import HMMRefiner
from rjmcmc_module import RJMCMCRefiner

class StateManagementToolkit:
    def __init__(self, config):
        self.config = config
        self.hmm_refiner = HMMRefiner(config)
        self.rjmcmc_refiner = RJMCMCRefiner(config)

    def refine_states(self, data):
        """Refine states using HMM and RJMCMC refiners."""
        print("Starting state refinement...")

        # HMM Refinement
        print("Applying HMM refinement...")
        data = self.hmm_refiner.refine_states(data)

        # RJMCMC Refinement
        print("Applying RJMCMC refinement...")
        data = self.rjmcmc_refiner.refine_states(data)

        print("State refinement complete.")
        return data

if __name__ == "__main__":
    print("This script provides a state management toolkit and should not be run independently.")
