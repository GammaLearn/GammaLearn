import os
from hipecta.data import PCalibRun
from hipecta.data import PSimulation
from hipecta.data import ctaTelescope2Matrix
from hipecta import core
import numpy as np
import h5py

import argparse

import hashlib

#import from hipecta.data import ctaloadfiles as cd

version = "1.0"

def loadCTAData(pruncalibfilename, simufilename):
    """
        Just load a calibrated prun file and a simulation file (should be related)
        Might be replaced later with generic functions from hipecta
        Parameters
        ----------
        pruncalibfilename: string
        simufilename: string

        Returns
        -------
        PCalibRun object
        Psimulation object
        sha1 string of the pcalibrun file name
        sha1 string of the psimu file name
        """
    pr = PCalibRun()
    pr.load(pruncalibfilename)
    ps = PSimulation()
    ps.load(simufilename)
    shaPr = hashlib.sha1(os.path.basename(pruncalibfilename).encode('utf8')).hexdigest()
    shaPs = hashlib.sha1(os.path.basename(simufilename).encode('utf8')).hexdigest()
    return pr, ps, shaPr, shaPs

def addDataToDataset(dataset,data):
    """
        Add data to an existing hdf5 resizable dataset
        Parameters
        ----------
        dataset: hdf5 dataset
        data: numpy array

        Returns
        -------
        dataset: hdf5 dataset
    """
    rowCount = dataset.shape[0]
    dataset.resize(rowCount+data.shape[0], axis=0)
    dataset[rowCount:]=data
    return dataset

# def fetchSimulationData(tabToFetch, itemToFetch):
#     simDict={}
#     for item in itemToFetch:
#         simDict[item] = np.array([eval('simu.'+item) for simu in tabToFetch])
#     return simDict

def CTAToHdf5( pruncalibfilename, simufilename, hdf5filename, telescopeTypeDict):
    """
        Convert telescope and simulation data to hdf5 and write them in a hdf5 file
        Might be replaced later with generic functions from hipecta
        Parameters
        ----------
        pr: PCalibRun object
        ps: PSimulation object
        hdf5filename: string
        telescopeTypeDict: dictionary

        """
    # Load Data
    pr, ps, shaPr, shaPs = loadCTAData(pruncalibfilename, simufilename)
    prfilename = os.path.basename(pruncalibfilename)
    psfilename = os.path.basename(simufilename)
    # Fetch data from pcalibrun & psimu files
    # Event simulation data
    eventIdSim = np.array([ev.id for ev in ps.tabSimuEvent])
    showerIdSim = np.array([ev.showerNum for ev in ps.tabSimuEvent])
    xCore = np.array([ev.xCore for ev in ps.tabSimuEvent])
    yCore = np.array([ev.yCore for ev in ps.tabSimuEvent])

    # Telescope direction
    # the same for each in simulation, should be modified for real data
    telAzimuth = pr.header.azimuth
    telAltitude = pr.header.altitude

    # Telescope data
    telescopeType = set([tel.telescopeType for tel in pr.tabTelescope])
    telDict = {}
    eventCounter = []
    for telType in telescopeType:
        telDict[telType] = {}
        # Telescope event data
        telDict[telType]['images'] = np.array([event.tabPixel for tel in pr.tabTelescope if tel.telescopeType == telType for event in tel.tabTelEvent])
        telDict[telType]['eventId'] = [event.eventId for tel in pr.tabTelescope if tel.telescopeType == telType for event in tel.tabTelEvent]
        eventCounter = eventCounter + telDict[telType]['eventId']
        telDict[telType]['eventId'] = np.array(telDict[telType]['eventId'])
        telDict[telType]['telescopeId'] =  np.array([tel.telescopeId for tel in pr.tabTelescope if tel.telescopeType == telType for event in tel.tabTelEvent])
        telDict[telType]['pixelsPosition'] = [tel.tabPos.tabPixelPosXY for tel in pr.tabTelescope if tel.telescopeType == telType][0]
        # Add matrix form images
        telDict[telType]['injTable'], telDict[telType]['nbRow'], telDict[telType]['nbCol'] = core.createAutoInjunctionTable(telDict[telType]['pixelsPosition'])
        # Add shower id
        telDict[telType]['showerId'] = np.squeeze(np.array([showerIdSim[np.where(eventIdSim==ev)] for ev in telDict[telType]['eventId']]))
        # Add telescope direction
        telDict[telType]['telescopeAltitude'] = np.array([telAltitude for tel in pr.tabTelescope if tel.telescopeType == telType for event in tel.tabTelEvent])
        telDict[telType]['telescopeAzimuth'] = np.array([telAzimuth for tel in pr.tabTelescope if tel.telescopeType == telType for event in tel.tabTelEvent])

    eventSet = set(eventCounter)

    # We only keep simulation data for detected event (telescope event)
    # Event simulation data
    idx = [np.where(eventIdSim == ev) for ev in eventSet]
    showerIdSim = showerIdSim[idx]
    showerIdSim = np.squeeze(showerIdSim)
    xCore = xCore[idx]
    xCore = np.squeeze(xCore)
    yCore = yCore[idx]
    yCore = np.squeeze(yCore)
    eventIdSim = eventIdSim[idx]
    eventIdSim = np.squeeze(eventIdSim)

    # Shower simulation data
    altitude = np.array([sh.altitude for sh in ps.tabSimuShower if sh.id in showerIdSim])
    azimuth = np.array([sh.azimuth for sh in ps.tabSimuShower if sh.id in showerIdSim])
    cmax = np.array([sh.cmax for sh in ps.tabSimuShower if sh.id in showerIdSim])
    depthStart = np.array([sh.depthStart for sh in ps.tabSimuShower if sh.id in showerIdSim])
    emax = np.array([sh.emax for sh in ps.tabSimuShower if sh.id in showerIdSim])
    energy = np.array([sh.energy for sh in ps.tabSimuShower if sh.id in showerIdSim])
    heightFirstInteraction = np.array([sh.heightFirstInteraction for sh in ps.tabSimuShower if sh.id in showerIdSim])
    hmax = np.array([sh.hmax for sh in ps.tabSimuShower if sh.id in showerIdSim])
    showerId = np.array([sh.id for sh in ps.tabSimuShower if sh.id in showerIdSim])
    particleType = np.array([sh.particleType for sh in ps.tabSimuShower if sh.id in showerIdSim])
    xmax = np.array([sh.xmax for sh in ps.tabSimuShower if sh.id in showerIdSim])

    # Telescope infos
    telId = np.array([tel.telescopeId for tel in pr.tabTelescope])
    telPosition = np.array([posTel for posTel in pr.header.tabPosTel])
    telFocal = np.array([focalTel for focalTel in pr.header.tabFocalTel])

    # Write data to hdf5 file
    # Prepare hdf5 object
    hdf5StuctureDict = {}

    if os.path.isfile(hdf5filename):
        gammaHdf = h5py.File(hdf5filename,'r+')
        # Load hdf5 file datasets
        fileConverterVersion = gammaHdf.attrs['converter_version']
        gammaHdf['pcalibrun_files'].attrs[shaPr]= prfilename
        gammaHdf['psimu_files'].attrs[shaPs]= psfilename
        # Telescope data
        for telType in telescopeType:
            telGrp = {}
            group = 'Cameras/' + telescopeTypeDict[telType]
            telGrp["showerId"] = gammaHdf[group + '/showerId']
            telGrp["showerId"] = addDataToDataset(telGrp["showerId"], telDict[telType]['showerId'])
            telGrp["images"] = gammaHdf[group + '/images']
            telGrp["images"] = addDataToDataset(telGrp["images"], telDict[telType]['images'])
            telGrp["eventId"] = gammaHdf[group + '/eventId']
            idStart = len(telGrp["eventId"])
            telGrp["eventId"] = addDataToDataset(telGrp["eventId"], telDict[telType]['eventId'])
            idEnd = len(telGrp["eventId"])-1
            telGrp["telescopeId"] = gammaHdf[group + '/telescopeId']
            telGrp["telescopeId"] = addDataToDataset(telGrp["telescopeId"], telDict[telType]['telescopeId'])
            telGrp["telescopeAltitude"] = gammaHdf[group + '/telescopeAltitude']
            telGrp["telescopeAltitude"] = addDataToDataset(telGrp["telescopeAltitude"], telDict[telType]['telescopeAltitude'])
            telGrp["telescopeAzimuth"] = gammaHdf[group + '/telescopeAzimuth']
            telGrp["telescopeAzimuth"] = addDataToDataset(telGrp["telescopeAzimuth"], telDict[telType]['telescopeAzimuth'])
            gammaHdf[group + "/pcalibrun_files"].attrs[shaPr] = [idStart, idEnd]
            hdf5StuctureDict[telType] = telGrp
        # Shower simulation data
        showerDataAltitude = gammaHdf['/showerSimu/altitude']
        showerDataAltitude = addDataToDataset(showerDataAltitude, altitude)
        showerDataAzimuth = gammaHdf['/showerSimu/azimuth']
        showerDataAzimuth = addDataToDataset(showerDataAzimuth, azimuth)
        showerDataCmax = gammaHdf['/showerSimu/cmax']
        showerDataCmax = addDataToDataset(showerDataCmax, cmax)
        showerDataDepthStart = gammaHdf['/showerSimu/depthStart']
        showerDataDepthStart = addDataToDataset(showerDataDepthStart, depthStart)
        showerDataEmax = gammaHdf['/showerSimu/emax']
        showerDataEmax = addDataToDataset(showerDataEmax, emax)
        showerDataEnergy = gammaHdf['/showerSimu/energy']
        showerDataEnergy = addDataToDataset(showerDataEnergy, energy)
        showerDataHeight = gammaHdf['/showerSimu/heightFirstInteraction']
        showerDataHeight = addDataToDataset(showerDataHeight, heightFirstInteraction)
        showerDataHmax = gammaHdf['/showerSimu/hmax']
        showerDataHmax = addDataToDataset(showerDataHmax, hmax)
        showerDataShowerId = gammaHdf['/showerSimu/showerId']
        idStartShower = len(showerDataShowerId)
        showerDataShowerId = addDataToDataset(showerDataShowerId,showerId)
        idEndShower = len(showerDataShowerId)-1
        showerDataXmax = gammaHdf['/showerSimu/xmax']
        showerDataXmax = addDataToDataset(showerDataXmax, xmax)
        gammaHdf['/showerSimu/psimu_files'].attrs[shaPs] = [idStartShower, idEndShower]
        # Event simulation data
        eventDataEventId = gammaHdf['/eventSimu/eventId']
        idStartEvent = len(eventDataEventId)
        eventDataEventId = addDataToDataset(eventDataEventId, eventIdSim)
        idEndEvent = len(eventDataEventId)-1
        eventDataShowerId = gammaHdf['/eventSimu/showerId']
        eventDataShowerId = addDataToDataset(eventDataShowerId, showerIdSim)
        eventDataXCore = gammaHdf['/eventSimu/xCore']
        eventDataXCore = addDataToDataset(eventDataXCore, xCore)
        eventDataYCore = gammaHdf['/eventSimu/yCore']
        eventDataYCore = addDataToDataset(eventDataYCore, yCore)
        gammaHdf['/eventSimu/psimu_files'].attrs[shaPs] = [idStartEvent, idEndEvent]
    else:
        gammaHdf = h5py.File(hdf5filename,'w')

        gammaHdf.attrs['particleType']=particleType[0]
        gammaHdf.attrs['converter_version']=version
        gammaHdf.attrs['HDF5_version']=h5py.version.hdf5_version
        gammaHdf.attrs['h5py_version']=h5py.version.version
        pcalibrun = gammaHdf.create_group('pcalibrun_files')
        pcalibrun.attrs[shaPr]=prfilename
        psimu = gammaHdf.create_group('psimu_files')
        psimu.attrs[shaPs]=psfilename

        # Create hdf5 structure with datasets
        # Telescope data
        cameras = gammaHdf.create_group('Cameras')
        for telType in telescopeType:
            telGrp = {}
            telGrp["group"] = cameras.create_group(telescopeTypeDict[telType])
            telGrp["files"] = telGrp["group"].create_group("pcalibrun_files")
            telGrp["group"].create_dataset("showerId",data=telDict[telType]['showerId'], maxshape=(None,),dtype=np.uint64)
            maxshape = (None,) + telDict[telType]['images'].shape[1:]
            telGrp["group"].create_dataset("images",data=telDict[telType]['images'], maxshape=maxshape,dtype=np.float32)
            telGrp["group"].create_dataset("eventId",data=telDict[telType]['eventId'],maxshape=(None,),dtype=np.uint64)
            telGrp["group"].create_dataset("telescopeId",data=telDict[telType]['telescopeId'],maxshape=(None,),dtype=np.uint64)
            telGrp["group"].create_dataset("telescopeAltitude", data=telDict[telType]['telescopeAltitude'], maxshape=(None,),dtype=np.float32)
            telGrp["group"]["telescopeAltitude"].attrs["units"] = "rad"
            telGrp["group"].create_dataset("telescopeAzimuth", data=telDict[telType]['telescopeAzimuth'], maxshape=(None,), dtype=np.float32)
            telGrp["group"]["telescopeAzimuth"].attrs["units"] = "rad"
            telGrp["group"].attrs["pixelsPosition"]=telDict[telType]['pixelsPosition']
            telGrp["group"].attrs["injTable"]=telDict[telType]['injTable']
            telGrp["group"].attrs["nbRow"]=telDict[telType]['nbRow']
            telGrp["group"].attrs["nbCol"]=telDict[telType]['nbCol']
            telGrp["files"].attrs[shaPr]=[0,len(telDict[telType]['eventId'])-1]
            hdf5StuctureDict[telType] = telGrp
        # Shower simulation data
        showerSimuGrp = gammaHdf.create_group("showerSimu")
        showerSimuGrp.create_group("psimu_files")
        showerDataAltitude = showerSimuGrp.create_dataset('altitude',data=altitude,maxshape=(None,),dtype=np.float32)
        showerDataAltitude.attrs["units"] = "rad"
        showerDataAzimuth = showerSimuGrp.create_dataset('azimuth',data=azimuth,maxshape=(None,),dtype=np.float32)
        showerDataAzimuth.attrs["units"] = "rad"
        showerSimuGrp.create_dataset('cmax',data=cmax,maxshape=(None,),dtype=np.float32)
        showerSimuGrp.create_dataset('depthStart',data=depthStart,maxshape=(None,),dtype=np.float32)
        showerSimuGrp.create_dataset('emax',data=emax,maxshape=(None,),dtype=np.float32)
        showerDataEnergy = showerSimuGrp.create_dataset('energy',data=energy,maxshape=(None,),dtype=np.float32)
        showerDataEnergy.attrs["units"] = "TeV"
        showerDataHeight = showerSimuGrp.create_dataset('heightFirstInteraction',data=heightFirstInteraction,maxshape=(None,),dtype=np.float32)
        showerDataHeight.attrs["units"] = "m"
        showerSimuGrp.create_dataset('hmax',data=hmax,maxshape=(None,),dtype=np.float32)
        showerSimuGrp.create_dataset('showerId',data=showerId,maxshape=(None,),dtype=np.uint64)
        showerSimuGrp.create_dataset('xmax',data=xmax,maxshape=(None,),dtype=np.float32)
        showerSimuGrp['psimu_files'].attrs[shaPs] = [0,len(showerId)]
        # Event simulation data
        eventSimuGrp = gammaHdf.create_group("eventSimu")
        eventSimuGrp.create_group("psimu_files")
        eventSimuGrp.create_dataset('eventId',data=eventIdSim,maxshape=(None,),dtype=np.uint64)
        eventSimuGrp.create_dataset('showerId',data=showerIdSim,maxshape=(None,),dtype=np.uint64)
        eventDataXCore = eventSimuGrp.create_dataset('xCore',data=xCore,maxshape=(None,),dtype=np.float32)
        eventDataXCore.attrs["units"] = "m"
        eventDataXCore.attrs["origin"] = "center of the site"
        eventDataYCore = eventSimuGrp.create_dataset('yCore',data=yCore,maxshape=(None,),dtype=np.float32)
        eventDataYCore.attrs["units"] = "m"
        eventDataYCore.attrs["origin"] = "center of the site"
        eventSimuGrp['psimu_files'].attrs[shaPs] = [0,len(eventIdSim)]
        # Telescope Infos
        telInfos = gammaHdf.create_group("telescopeInfos")
        telInfos.create_group("pcalibrun_files")
        telInfos.create_dataset('telescopeId',data=telId,dtype=np.uint64)
        telDataPosition = telInfos.create_dataset('telescopePosition',data=telPosition,dtype=np.float32)
        telDataPosition.attrs["units"] = "m"
        telDataPosition.attrs["origin"] = "center of the site"
        telInfos.create_dataset('telescopeFocal',data=telFocal,dtype=np.float32)
        telInfos['pcalibrun_files'].attrs[shaPr] = [0, len(telId)]

    gammaHdf.close()


def squeezeData(l):
    """
    Squeeze an array or a list of length 1
    Parameters
    ----------
    l: np.ndarray or list

    Returns
    -------
    variable
    """
    while isinstance(l,(np.ndarray,list)):
        if len(l)==1:
            l = l[0]
        else:
            break

    return l

def extractRandomImageDataFromHDF5(hdf5filename,telescopeTypeDict):
    """
    Extract all the data linked to a random image from a random type of telescope
    Parameters
    ----------
    hdf5filename: string
    telescopeTypeDict: dictionary

    Returns
    -------
    dictionary
    """
    with h5py.File(hdf5filename,'r') as f:
        eventData = {}
        camNum = np.random.choice(len(f['Cameras'].keys()))
        camType = list(f['Cameras'].keys())[camNum]
        cam = 'Cameras/' + camType
        eventIndex = np.random.choice(len(f[cam+'/eventId']))
        shaPr = [attr for attr in f[cam+'/pcalibrun_files'].attrs if (eventIndex>=f[cam+'/pcalibrun_files'].attrs[attr][0])&(eventIndex<=f[cam+'/pcalibrun_files'].attrs[attr][1])]
        eventData['pcalibrun'] = f['/pcalibrun_files'].attrs[squeezeData(shaPr)]
        eventData['telescopeType'] = list(telescopeTypeDict.keys())[list(telescopeTypeDict.values()).index(camType)]
        eventData['eventId'] = f[cam+'/eventId'][eventIndex]
        eventData['showerId'] = f[cam+'/showerId'][eventIndex]
        showerIndex = np.where(np.array(f['showerSimu/showerId']) == eventData['showerId'])
        showerIndex = squeezeData(list(showerIndex))
        shaPs = [attr for attr in f['showerSimu/psimu_files'].attrs if
                 (showerIndex >= f['showerSimu/psimu_files'].attrs[attr][0]) & (showerIndex <= f['showerSimu/psimu_files'].attrs[attr][1])]
        eventData['psimu'] = f['psimu_files'].attrs[squeezeData(shaPs)]
        eventData['image'] = f[cam+'/images'][eventIndex]
        eventData['telescopeAltitude'] = f[cam + '/telescopeAltitude'][eventIndex]
        eventData['telescopeAzimuth'] = f[cam + '/telescopeAzimuth'][eventIndex]
        eventData['telescopeId'] = f[cam + '/telescopeId'][eventIndex]
        eventData['pixelsPosition'] = np.squeeze(f[cam].attrs['pixelsPosition'])
        eventData['xCore'] = f['eventSimu/xCore'][(np.array(f['eventSimu/showerId'])==eventData['showerId']) & (np.array(f['eventSimu/eventId'])==eventData['eventId'])]
        eventData['yCore'] = f['eventSimu/yCore'][(np.array(f['eventSimu/showerId'])==eventData['showerId']) & (np.array(f['eventSimu/eventId'])==eventData['eventId'])]
        eventData['hmax'] = f['showerSimu/hmax'][showerIndex]
        eventData['xmax'] = f['showerSimu/xmax'][showerIndex]
        eventData['cmax'] = f['showerSimu/cmax'][showerIndex]
        eventData['azimuth'] = f['showerSimu/azimuth'][showerIndex]
        eventData['altitude'] = f['showerSimu/altitude'][showerIndex]
        eventData['energy'] = f['showerSimu/energy'][showerIndex]
        eventData['heightFirstInteraction'] = f['showerSimu/heightFirstInteraction'][showerIndex]
        eventData['emax'] = f['showerSimu/emax'][showerIndex]
        eventData['depthStart'] = f['showerSimu/depthStart'][showerIndex]
        eventData['telescopePosition'] = f['telescopeInfos/telescopePosition'][np.array(f['telescopeInfos/telescopeId']) == eventData['telescopeId'],:]
        eventData['telescopeFocal'] = f['telescopeInfos/telescopeFocal'][np.array(f['telescopeInfos/telescopeId']) == eventData['telescopeId']]

        for key in list(eventData.keys()):
            eventData[key]=squeezeData(eventData[key])
        return eventData

def extractImageDataFromPcalibrun(dataExtractedFromHdf5,pathTofiles):
    """
    Extract from origin pcalibrun and psim files the data related to those extracted from hdf5
    Then compare them
    Parameters
    ----------
    dataExtractedFromHdf5: dictionary

    Returns
    -------

    """
    # Extract data
    eventData = {}
    eventData['pcalibrun']= dataExtractedFromHdf5['pcalibrun']
    eventData['psimu'] = dataExtractedFromHdf5['psimu']

    pr, ps, shaPr, shaPs = loadCTAData(pathTofiles+eventData['pcalibrun'],pathTofiles+eventData['psimu'])

    eventData['eventId'] = dataExtractedFromHdf5['eventId']
    eventData['showerId'] = dataExtractedFromHdf5['showerId']
    eventData['telescopeId'] = dataExtractedFromHdf5['telescopeId']
    eventData['image'] = [ev.tabPixel for tel in pr.tabTelescope if tel.telescopeId==eventData['telescopeId'] for ev in tel.tabTelEvent if ev.eventId==eventData['eventId']]
    eventData['pixelsPosition'] = np.squeeze([tel.tabPos.tabPixelPosXY for tel in pr.tabTelescope if tel.telescopeId == eventData['telescopeId']])
    eventData['telescopeType'] = [tel.telescopeType for tel in pr.tabTelescope if tel.telescopeId==eventData['telescopeId']]
    eventData['telescopeAltitude'] = pr.header.altitude
    eventData['telescopeAzimuth'] = pr.header.azimuth
    eventData['xCore'] = [ev.xCore for ev in ps.tabSimuEvent if (ev.id==eventData['eventId']) & (ev.showerNum==eventData['showerId'])]
    eventData['yCore'] = [ev.yCore for ev in ps.tabSimuEvent if (ev.id==eventData['eventId']) & (ev.showerNum==eventData['showerId'])]
    eventData['hmax'] = [sh.hmax for sh in ps.tabSimuShower if sh.id==eventData['showerId']]
    eventData['xmax'] = [sh.xmax for sh in ps.tabSimuShower if sh.id==eventData['showerId']]
    eventData['cmax'] = [sh.cmax for sh in ps.tabSimuShower if sh.id==eventData['showerId']]
    eventData['azimuth'] = [sh.azimuth for sh in ps.tabSimuShower if sh.id==eventData['showerId']]
    eventData['altitude'] = [sh.altitude for sh in ps.tabSimuShower if sh.id==eventData['showerId']]
    eventData['energy'] = [sh.energy for sh in ps.tabSimuShower if sh.id==eventData['showerId']]
    eventData['heightFirstInteraction'] = [sh.heightFirstInteraction for sh in ps.tabSimuShower if sh.id==eventData['showerId']]
    eventData['emax'] = [sh.emax for sh in ps.tabSimuShower if sh.id==eventData['showerId']]
    eventData['depthStart'] = [sh.depthStart for sh in ps.tabSimuShower if sh.id==eventData['showerId']]
    telId = np.array([tel.telescopeId for tel in pr.tabTelescope])
    telPosition = np.array([posTel for posTel in pr.header.tabPosTel])
    telFocal = np.array([focalTel for focalTel in pr.header.tabFocalTel])
    eventData['telescopePosition'] = np.squeeze(telPosition[np.where(telId==eventData['telescopeId'])])
    eventData['telescopeFocal'] = telFocal[np.where(telId==eventData['telescopeId'])]

    for key in list(eventData.keys()):
        eventData[key] = squeezeData(eventData[key])

    # Compare data
    for key in list(dataExtractedFromHdf5.keys()):
        if isinstance(eventData[key],np.ndarray):
            res = np.array_equal(eventData[key] , dataExtractedFromHdf5[key])
        else:
            res = eventData[key] == dataExtractedFromHdf5[key]
        if res:
            print(key + ' matches')
        else:
            print(eventData[key])
            print(dataExtractedFromHdf5[key])
            print(key + ' doesn\'t match, verification failed !')
            break

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

# Example of use. Need to define the way to get the file names
# pruncalibfilename = '/home/jacquemont/projets_CTA/Prod3b/Paranal/Gamma_point_source/gamma_20deg_0deg_run4514___cta-prod3_desert-2150m-Paranal-merged.pcalibRun'
# simufilename = '/home/jacquemont/projets_CTA/Prod3b/Paranal/Gamma_point_source/gamma_20deg_0deg_run4514___cta-prod3_desert-2150m-Paranal-merged.psimu'
# hdf5filename = '/home/jacquemont/projets_CTA/gamma.hdf5'

parser = argparse.ArgumentParser()
parser.add_argument("pcalibrunfile", help="pcalibrun file name to read")
parser.add_argument("psimulation", help="psimulation file name to read")
parser.add_argument("hdf5file", help="hdf5 file name to write")
args = parser.parse_args()
pruncalibfilename = args.pcalibrunfile
simufilename = args.psimulation
hdf5filename = args.hdf5file

telescopeTypeDict = {0:'DRAGON', 1:'NECTAR', 2:'FLASH', 3:'SCT', 4:'ASTRI', 5:'DC', 6:'GCT'}

#CTAToHdf5(pruncalibfilename, simufilename, hdf5filename, telescopeTypeDict)
dic=extractRandomImageDataFromHDF5(hdf5filename,telescopeTypeDict)
pathToFiles = '/home/jacquemont/projets_CTA/Prod3b/Paranal/Gamma_point_source/'
extractImageDataFromPcalibrun(dic,pathToFiles)