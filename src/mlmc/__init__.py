"""
Classes:
mlmc.MLMC - Used to generate samples, possibly estimate required number of samples on levels
            using given estimates for diff variances. Create log of running simulations.
mlmc.Simulation - Base class for a simulation.
mlmc.Level - Internal class, may turn out unnecessary.
mlmc.Samples - Class storing all samples and providing various estimate functions for means and variances
               for given moments. Allow
mlmc.Distribution - Class providing estimate of whole distribution for given instance of Samples class.
               Can be used incrementally using previous PDF and domain for building the final PDF.
mlmc.LevelSamples - Internal class. possibly just a np.array with samples
"""