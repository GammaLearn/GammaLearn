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






def energyband_extractor(prunfilename):
    """

    Parameters
    ----------
    prunfilename

    Returns
    -------

    """