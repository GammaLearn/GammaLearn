import os
from hipecta.data import PCalibRun
from hipecta.data import PSimulation
from hipecta.data import ctaTelescope2Matrix
from hipecta import core
import numpy as np
import h5py

#import from hipecta.data import ctaloadfiles as cd


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
		"""
	pr = PCalibRun()
	pr.load(pruncalibfilename)
	ps = PSimulation()
	ps.load(simufilename)
	return pr, ps

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

def CTAToHdf5(pr, ps, hdf5filename, telescopeTypeDict):
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

	# Fetch data from pcalibrun & psimu files
	# Shower simulation data
	altitude = np.array([sh.altitude for sh in ps.tabSimuShower])
	azimuth = np.array([sh.azimuth for sh in ps.tabSimuShower])
	cmax = np.array([sh.cmax for sh in ps.tabSimuShower])
	depthStart = np.array([sh.depthStart for sh in ps.tabSimuShower])
	emax = np.array([sh.emax for sh in ps.tabSimuShower])
	energy = np.array([sh.energy for sh in ps.tabSimuShower])
	heightFirstInteraction = np.array([sh.heightFirstInteraction for sh in ps.tabSimuShower])
	hmax = np.array([sh.hmax for sh in ps.tabSimuShower])
	showerId = np.array([sh.id for sh in ps.tabSimuShower])
	particleType = np.array([sh.particleType for sh in ps.tabSimuShower])
	xmax = np.array([sh.xmax for sh in ps.tabSimuShower])

	# Event simulation data
	eventIdSim = np.array([ev.id for ev in ps.tabSimuEvent])
	showerIdSim = np.array([ev.showerNum for ev in ps.tabSimuEvent])
	xCore = np.array([ev.xCore for ev in ps.tabSimuEvent])
	yCore = np.array([ev.yCore for ev in ps.tabSimuEvent])

	# Telescope infos
	telId = np.array([tel.telescopeId for tel in pr.tabTelescope])
	telPosition = np.array([posTel for posTel in pr.header.tabPosTel])
	telFocal = np.array([focalTel for focalTel in pr.header.tabFocalTel])

	# Telescope data
	telescopeType = set([tel.telescopeType for tel in pr.tabTelescope])
	telDict = {}
	for telType in telescopeType:
		telDict[telType] = {}
		# Telescope event data
		telDict[telType]['images'] = np.array([event.tabPixel for tel in pr.tabTelescope if tel.telescopeType == telType for event in tel.tabTelEvent])
		telDict[telType]['eventId'] = np.array([event.eventId for tel in pr.tabTelescope if tel.telescopeType == telType for event in tel.tabTelEvent])
		telDict[telType]['telescopeId'] =  np.array([tel.telescopeId for tel in pr.tabTelescope if tel.telescopeType == telType for event in tel.tabTelEvent])
		telDict[telType]['pixelsPosition'] = [tel.tabPos.tabPixelPosXY for tel in pr.tabTelescope if tel.telescopeType == telType][0]
		# Add matrix form images
		telDict[telType]['injTable'], telDict[telType]['nbRow'], telDict[telType]['nbCol'] = core.createAutoInjunctionTable(telDict[telType]['pixelsPosition'])
		# Add shower id
		telDict[telType]['showerId'] = np.array([showerIdSim[np.where(eventIdSim==ev)] for ev in telDict[telType]['eventId']])

	# Write data to hdf5 file
	# Prepare hdf5 object
	hdf5StuctureDict = {}

	if os.path.isfile(hdf5filename):
		gammaHdf = h5py.File(hdf5filename,'r+')
		# Load hdf5 file datasets
		# Telescope data
		for telType in telescopeType:
			telGrp = {}
			#telGrp["group"] = gammaHdf[telescopeTypeDict[telType]]
			group = telescopeTypeDict[telType]
			telGrp["showerId"] = gammaHdf[group + '/showerId']
			telGrp["showerId"] = addDataToDataset(telGrp["showerId"], telDict[telType]['showerId'])
			telGrp["images"] = gammaHdf[group + '/images']
			telGrp["images"] = addDataToDataset(telGrp["images"], telDict[telType]['images'])
			#telGrp["pixelsPosition"] = gammaHdf[group + '/pixelsPosition']
			telGrp["eventId"] = gammaHdf[group + '/eventId']
			telGrp["eventId"] = addDataToDataset(telGrp["eventId"], telDict[telType]['eventId'])
			telGrp["telescopeId"] = gammaHdf[group + '/telescopeId']
			telGrp["telescopeId"] = addDataToDataset(telGrp["telescopeId"], telDict[telType]['telescopeId'])
			#telGrp["injTable"] = gammaHdf[group + '/injTable']
			#telGrp["nbRow"] = gammaHdf[group + '/nbRow']
			#telGrp["nbCol"] = gammaHdf[group + '/nbCol']
			hdf5StuctureDict[telType] = telGrp
		# Shower simulation data
		#showerSimuGrp = gammaHdf.create_group("showerSimu")
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
		showerDataShowerId = addDataToDataset(showerDataShowerId,showerId)
		showerDataParticleType = gammaHdf['/showerSimu/particleType']
		showerDataParticleType = addDataToDataset(showerDataParticleType, particleType)
		showerDataXmax = gammaHdf['/showerSimu/xmax']
		showerDataXmax = addDataToDataset(showerDataXmax, xmax)
		# Event simulation data
		#eventSimuGrp = gammaHdf.create_group("eventSimu")
		eventDataEventId = gammaHdf['/eventSimu/eventId']
		eventDataEventId = addDataToDataset(eventDataEventId, eventIdSim)
		eventDataShowerId = gammaHdf['/eventSimu/showerId']
		eventDataShowerId = addDataToDataset(eventDataShowerId, showerIdSim)
		eventDataXCore = gammaHdf['/eventSimu/xCore']
		eventDataXCore = addDataToDataset(eventDataXCore, xCore)
		eventDataYCore = gammaHdf['/eventSimu/yCore']
		eventDataYCore = addDataToDataset(eventDataYCore, yCore)
		# Telescope Infos
		#telInfos = gammaHdf.create_group("telescopeInfos")
		#telescopeInfoId = gammaHdf['/telescopeInfos/telescopeId']
		#telescopeInfoPosition = gammaHdf['/telescopeInfos/telescopePosition']
		#telescopeInfoFocal = gammaHdf['/telescopeInfos/telescopeFocal']
	else:
		gammaHdf = h5py.File(hdf5filename,'w')
		# Create hdf5 structure with datasets
		# Telescope data
		for telType in telescopeType:
			telGrp = {}
			telGrp["group"] = gammaHdf.create_group(telescopeTypeDict[telType])
			telGrp["showerId"] = telGrp["group"].create_dataset("showerId",data=telDict[telType]['showerId'], maxshape=(None,1),dtype=np.uint64)
			maxshape = (None,) + telDict[telType]['images'].shape[1:]
			telGrp["images"] = telGrp["group"].create_dataset("images",data=telDict[telType]['images'], maxshape=maxshape,dtype=np.float32)
			telGrp["pixelsPosition"] = telGrp["group"].create_dataset("pixelsPosition",data=telDict[telType]['pixelsPosition'],dtype=np.float32)
			telGrp["eventId"] = telGrp["group"].create_dataset("eventId",data=telDict[telType]['eventId'],maxshape=(None,),dtype=np.uint64)
			telGrp["telescopeId"] = telGrp["group"].create_dataset("telescopeId",data=telDict[telType]['telescopeId'],maxshape=(None,),dtype=np.uint64)
			telGrp["injTable"] = telGrp["group"].create_dataset("injTable",data=telDict[telType]['injTable'],dtype=np.uint64)
			telGrp["nbRow"] = telGrp["group"].create_dataset("nbRow",data=telDict[telType]['nbRow'],dtype=np.int8)
			telGrp["nbCol"] = telGrp["group"].create_dataset("nbCol",data=telDict[telType]['nbCol'],dtype=np.int8)
			hdf5StuctureDict[telType] = telGrp
		# Shower simulation data
		showerSimuGrp = gammaHdf.create_group("showerSimu")
		showerDataAltitude = showerSimuGrp.create_dataset('altitude',data=altitude,maxshape=(None,),dtype=np.float32)
		showerDataAzimuth = showerSimuGrp.create_dataset('azimuth',data=azimuth,maxshape=(None,),dtype=np.float32)
		showerDataCmax = showerSimuGrp.create_dataset('cmax',data=cmax,maxshape=(None,),dtype=np.float32)
		showerDataDepthStart = showerSimuGrp.create_dataset('depthStart',data=depthStart,maxshape=(None,),dtype=np.float32)
		showerDataEmax = showerSimuGrp.create_dataset('emax',data=emax,maxshape=(None,),dtype=np.float32)
		showerDataEnergy = showerSimuGrp.create_dataset('energy',data=energy,maxshape=(None,),dtype=np.float32)
		showerDataHeight = showerSimuGrp.create_dataset('heightFirstInteraction',data=heightFirstInteraction,maxshape=(None,),dtype=np.float32)
		showerDataHmax = showerSimuGrp.create_dataset('hmax',data=hmax,maxshape=(None,),dtype=np.float32)
		showerDataShowerId = showerSimuGrp.create_dataset('showerId',data=showerId,maxshape=(None,),dtype=np.uint64)
		showerDataParticleType = showerSimuGrp.create_dataset('particleType',data=particleType,maxshape=(None,),dtype=np.int32)
		showerDataXmax = showerSimuGrp.create_dataset('xmax',data=xmax,maxshape=(None,),dtype=np.float32)
		# Event simulation data
		eventSimuGrp = gammaHdf.create_group("eventSimu")
		eventDataEventId = eventSimuGrp.create_dataset('eventId',data=eventIdSim,maxshape=(None,),dtype=np.uint64)
		eventDataShowerId = eventSimuGrp.create_dataset('showerId',data=showerIdSim,maxshape=(None,),dtype=np.uint64)
		eventDataXCore = eventSimuGrp.create_dataset('xCore',data=xCore,maxshape=(None,),dtype=np.float32)
		eventDataYCore = eventSimuGrp.create_dataset('yCore',data=yCore,maxshape=(None,),dtype=np.float32)
		# Telescope Infos
		telInfos = gammaHdf.create_group("telescopeInfos")
		telescopeInfoId = telInfos.create_dataset('telescopeId',data=telId,dtype=np.uint64)
		telescopeInfoPosition = telInfos.create_dataset('telescopePosition',data=telPosition,dtype=np.float32)
		telescopeInfoFocal = telInfos.create_dataset('telescopeFocal',data=telFocal,dtype=np.float32)

	gammaHdf.close()





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

	return np.array([event.tabPixel for tel in pr.tabTelescope if tel.telescopeType == telescope_type for event in tel.tabTelEvent])




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
pruncalibfilename = '/home/jacquemont/projets_CTA/Prod3b/Paranal/Gamma_point_source/gamma_20deg_0deg_run4514___cta-prod3_desert-2150m-Paranal-merged.pcalibRun'
simufilename = '/home/jacquemont/projets_CTA/Prod3b/Paranal/Gamma_point_source/gamma_20deg_0deg_run4514___cta-prod3_desert-2150m-Paranal-merged.psimu'
hdf5filename = '/home/jacquemont/projets_CTA/gamma.hdf5'
telescopeTypeDict = {0:'DRAGON', 1:'NECTAR', 2:'FLASH', 3:'SCT', 4:'ASTRI', 5:'DC', 6:'GCT'}

pr, ps = loadCTAData(pruncalibfilename, simufilename)
CTAToHdf5(pr, ps, hdf5filename, telescopeTypeDict)