# Package to make HDF5 files out of prun files



____
--TODO list--

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


____
--HF5 files structure:--  

1 file = 1 particle type

physical parameters = tensor
- ShowerID, E, Alt, Az, Hmax, atmDepth, atmosphere, impact point
- attributes (dict of the parameters) = list of string

showerId = key

groups + datasets

e.g. :
- group = camera type
- sub-group mandatory:
    - pixels position
    - images
    - EventIds = 1D array
    - showerIndex
    - telescopeID
- sub-group optional:
    - rectangular pixels positions
