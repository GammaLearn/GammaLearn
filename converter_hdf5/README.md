# Package to make HDF5 files out of prun files



____
**TODO list**

- function to extract different kinds of selected images from prun files
    - telescope
    - simulated energy
    - amplitude
    
- format to extract? Numpy arrays?
    - 4D (3D for integrated images) Numpy arrays
    - array of 3D (2D) RAW data
    - other arrays with simulation info
        - energies
        - telescope id
    - header with
        - telescope types for each tel id
        - telescope position for each tel id