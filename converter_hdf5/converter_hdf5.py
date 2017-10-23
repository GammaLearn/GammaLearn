import numpy
import ctadata as cd

# Will become:
# from hipecta import ctaloadfiles
# from hipecta.ctaloadfiles import load_ctafile

from ctaDataWrapper import PCalibRun


def load_calibrated_prun(filename):
    """
    Just load a calibrated prun file
    Might be replaced later with generic functions from hipecta
    Parameters
    ----------
    filename: string

    Returns
    -------
    prun object
    """
    pr = PCalibRun()
    pr.load(pruncalibfilename)
    return pr


def extract_images_telescope(pcalibrun_filename, telescope_type):
    """
    Extract images from a calibrated file for a unique telescope type
    and stack them in a numpy array
    Parameters
    ----------
    pcalibrun_filename: string
    telescope_type: int

    Returns
    -------
    2D Numpy array
    """
    pr = PCalibRun()
    pr.load(pcalibrun_filename)

    return np.array([event.tabPixel for tel in pr.tabTelescope for event in tel.tabTelEvent if tel.telescopeType == telescope_type])





def energyband_extractor(pcalibrun_filename, psimu_filename, emin, emax):

    """
    Extract images from a pcalibrun file for events in a given energy band.
    Simulated energy is given by the psimu file
    Return a dictionnary with stacked images per telescope type

    Parameters
    ----------
    pcalibrun_filename: string
    psimu_filename: string
    emin: float
    emax: float

    Returns
    -------

    """