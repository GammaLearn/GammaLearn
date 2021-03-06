import os
from hipecta.data import PCalibRun
from hipecta.data import PSimulation
from hipecta import core
import numpy as np
import h5py

import sys

import re


_VERSION = "1.1"
_TELESCOPE_TYPE_DICT = {0:'LSTCAM', 1:'NECTAR', 2:'FLASH', 3:'SCT', 4:'ASTRI', 5:'DC', 6:'GCT'}

def load_cta_data(pruncalibfilename, simufilename):
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

    return pr, ps

def add_data_to_dataset(dataset,data):
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
    row_count = dataset.shape[0]
    dataset.resize(row_count+data.shape[0], axis=0)
    dataset[row_count:]=data
    return dataset

# def fetchSimulationData(tabToFetch, itemToFetch):
#     simDict={}
#     for item in itemToFetch:
#         simDict[item] = np.array([eval('simu.'+item) for simu in tabToFetch])
#     return simDict

def cta_to_hdf5( pruncalibfilename, simufilename, hdf5filename):
    """
        Convert telescope and simulation data to hdf5 and write them in a hdf5 file
        Might be replaced later with generic functions from hipecta
        Parameters
        ----------
        pr: string
        ps: string
        hdf5filename: string

        """
    # Load Data
    pr, ps = load_cta_data(pruncalibfilename, simufilename)
    prfilename = os.path.basename(pruncalibfilename)
    psfilename = os.path.basename(simufilename)

    #Extract run id
    word = re.search("_run(\d+)", prfilename)
    if word:
        run_id = int(word.group(1))
        print("run id : ", run_id)
    else:
        # exit("Didn't find run id for file : " + prfilename)
        run_id = 0

    # Check if pcalibrun file is already in hdf5
    if os.path.isfile(hdf5filename):
        hdf5_file = h5py.File(hdf5filename,'r')
        if str(run_id) in set(hdf5_file['pcalibrun_files'].attrs):
            print("File %s already converted. Skipping" % prfilename)
            return
        hdf5_file.close()

    # Fetch data from pcalibrun & psimu files
    # Event simulation data
    event_id_sim = np.array([ev.id for ev in ps.tabSimuEvent])
    shower_id_sim = np.array([ev.showerNum for ev in ps.tabSimuEvent])
    xCore = np.array([ev.xCore for ev in ps.tabSimuEvent])
    yCore = np.array([ev.yCore for ev in ps.tabSimuEvent])

    # Telescope direction
    # the same for each in simulation, should be modified for real data
    tel_azimuth = pr.header.azimuth
    tel_altitude = pr.header.altitude

    # Telescope data
    telescope_type = set([tel.telescopeType for tel in pr.tabTelescope])
    tel_dict = {}
    event_counter = []
    for tel_type in telescope_type:
        tel_dict[tel_type] = {}
        # Telescope event data
        tel_dict[tel_type]['images'] = np.array([event.tabPixel for tel in pr.tabTelescope if tel.telescopeType == tel_type for event in tel.tabTelEvent])
        tel_dict[tel_type]['eventId'] = [event.eventId for tel in pr.tabTelescope if tel.telescopeType == tel_type for event in tel.tabTelEvent]
        event_counter = event_counter + tel_dict[tel_type]['eventId']
        tel_dict[tel_type]['eventId'] = np.array(tel_dict[tel_type]['eventId'])
        tel_dict[tel_type]['runId'] = np.full((len(tel_dict[tel_type]['eventId'])), run_id)
        tel_dict[tel_type]['telescopeId'] =  np.array([tel.telescopeId for tel in pr.tabTelescope if tel.telescopeType == tel_type for event in tel.tabTelEvent])
        tel_dict[tel_type]['pixelsPosition'] = [tel.tabPos.tabPixelPosXY for tel in pr.tabTelescope if tel.telescopeType == tel_type][0]
        # Add matrix form images
        tel_dict[tel_type]['injTable'], tel_dict[tel_type]['nbRow'], tel_dict[tel_type]['nbCol'] = core.createAutoInjunctionTable(tel_dict[tel_type]['pixelsPosition'])
        # Add telescope direction
        tel_dict[tel_type]['telescopeAltitude'] = np.array([tel_altitude for tel in pr.tabTelescope if tel.telescopeType == tel_type for event in tel.tabTelEvent])
        tel_dict[tel_type]['telescopeAzimuth'] = np.array([tel_azimuth for tel in pr.tabTelescope if tel.telescopeType == tel_type for event in tel.tabTelEvent])

    event_set = set(event_counter)

    # We only keep simulation data for detected event (telescope event)
    # Event simulation data
    event_triggered_id = np.array([ev.eventId for tel in pr.tabTelescope for ev in tel.tabTelEvent])
    idx = np.in1d(event_id_sim, event_triggered_id)
    shower_id_sim = shower_id_sim[idx]
    shower_id_sim = np.squeeze(shower_id_sim)
    xCore = xCore[idx]
    xCore = np.squeeze(xCore)
    yCore = yCore[idx]
    yCore = np.squeeze(yCore)
    event_id_sim = event_id_sim[idx]
    event_id_sim = np.squeeze(event_id_sim)
    run_id_event = np.full((len(event_id_sim)), run_id)

    # Shower simulation data
    altitude = np.array([sh.altitude for sh in ps.tabSimuShower if sh.id in shower_id_sim])
    azimuth = np.array([sh.azimuth for sh in ps.tabSimuShower if sh.id in shower_id_sim])
    cmax = np.array([sh.cmax for sh in ps.tabSimuShower if sh.id in shower_id_sim])
    depthStart = np.array([sh.depthStart for sh in ps.tabSimuShower if sh.id in shower_id_sim])
    emax = np.array([sh.emax for sh in ps.tabSimuShower if sh.id in shower_id_sim])
    energy = np.array([sh.energy for sh in ps.tabSimuShower if sh.id in shower_id_sim])
    heightFirstInteraction = np.array([sh.heightFirstInteraction for sh in ps.tabSimuShower if sh.id in shower_id_sim])
    hmax = np.array([sh.hmax for sh in ps.tabSimuShower if sh.id in shower_id_sim])
    showerId = np.array([sh.id for sh in ps.tabSimuShower if sh.id in shower_id_sim])
    particleType = np.array([sh.particleType for sh in ps.tabSimuShower if sh.id in shower_id_sim])
    xmax = np.array([sh.xmax for sh in ps.tabSimuShower if sh.id in shower_id_sim])
    run_id_shower = np.full((len(showerId)), run_id)

    # Telescope infos
    tel_id = np.array([tel.telescopeId for tel in pr.tabTelescope])
    tel_position = np.array([posTel for posTel in pr.header.tabPosTel])
    tel_focal = np.array([focalTel for focalTel in pr.header.tabFocalTel])

    # Write data to hdf5 file
    if os.path.isfile(hdf5filename):
        hdf5_file = h5py.File(hdf5filename,'r+')
        # Load hdf5 file datasets
        file_converter_version = hdf5_file.attrs['converter_version']
        if file_converter_version != _VERSION:
            sys.exit("Existing hdf5 file : converter version dosen't match !")
        hdf5_file['pcalibrun_files'].attrs[str(run_id)] = prfilename
        hdf5_file['psimu_files'].attrs[str(run_id)] = psfilename
        # Telescope data
        for tel_type in telescope_type:
            group = 'Cameras/' + _TELESCOPE_TYPE_DICT[tel_type]
            add_data_to_dataset(hdf5_file[group + '/runId'], tel_dict[tel_type]['runId'])
            add_data_to_dataset(hdf5_file[group + '/images'], tel_dict[tel_type]['images'])
            add_data_to_dataset( hdf5_file[group + '/eventId'], tel_dict[tel_type]['eventId'])
            add_data_to_dataset(hdf5_file[group + '/telescopeId'], tel_dict[tel_type]['telescopeId'])
            add_data_to_dataset(hdf5_file[group + '/telescopeAltitude'], tel_dict[tel_type]['telescopeAltitude'])
            add_data_to_dataset(hdf5_file[group + '/telescopeAzimuth'], tel_dict[tel_type]['telescopeAzimuth'])
        # Shower simulation data
        add_data_to_dataset(hdf5_file['/showerSimu/altitude'], altitude)
        add_data_to_dataset(hdf5_file['/showerSimu/azimuth'], azimuth)
        add_data_to_dataset(hdf5_file['/showerSimu/cmax'], cmax)
        add_data_to_dataset(hdf5_file['/showerSimu/depthStart'], depthStart)
        add_data_to_dataset(hdf5_file['/showerSimu/emax'], emax)
        add_data_to_dataset(hdf5_file['/showerSimu/energy'], energy)
        add_data_to_dataset(hdf5_file['/showerSimu/heightFirstInteraction'], heightFirstInteraction)
        add_data_to_dataset(hdf5_file['/showerSimu/hmax'], hmax)
        add_data_to_dataset(hdf5_file['/showerSimu/showerId'],showerId)
        add_data_to_dataset(hdf5_file['/showerSimu/runId'], run_id_shower)
        add_data_to_dataset(hdf5_file['/showerSimu/xmax'], xmax)
        # Event simulation data
        add_data_to_dataset(hdf5_file['/eventSimu/eventId'], event_id_sim)
        add_data_to_dataset(hdf5_file['/eventSimu/showerId'], shower_id_sim)
        add_data_to_dataset(hdf5_file['/eventSimu/runId'], run_id_event)
        add_data_to_dataset(hdf5_file['/eventSimu/xCore'], xCore)
        add_data_to_dataset(hdf5_file['/eventSimu/yCore'], yCore)
    else:
        hdf5_file = h5py.File(hdf5filename,'w')

        hdf5_file.attrs['particleType'] = particleType[0]
        hdf5_file.attrs['converter_version'] = _VERSION
        hdf5_file.attrs['HDF5_version'] = h5py.version.hdf5_version
        hdf5_file.attrs['h5py_version'] = h5py.version.version
        pcalibrun = hdf5_file.create_group('pcalibrun_files')
        pcalibrun.attrs[str(run_id)] = prfilename
        psimu = hdf5_file.create_group('psimu_files')
        psimu.attrs[str(run_id)] = psfilename

        # Create hdf5 structure with datasets
        # Telescope data
        cameras = hdf5_file.create_group('Cameras')
        for tel_type in telescope_type:
            grp = cameras.create_group(_TELESCOPE_TYPE_DICT[tel_type])
            grp.create_dataset("runId",data=tel_dict[tel_type]['runId'], maxshape=(None,),dtype=np.uint64)
            maxshape = (None,) + tel_dict[tel_type]['images'].shape[1:]
            grp.create_dataset("images",data=tel_dict[tel_type]['images'], maxshape=maxshape,dtype=np.float32)
            grp.create_dataset("eventId",data=tel_dict[tel_type]['eventId'],maxshape=(None,),dtype=np.uint64)
            grp.create_dataset("telescopeId",data=tel_dict[tel_type]['telescopeId'],maxshape=(None,),dtype=np.uint64)
            grp.create_dataset("telescopeAltitude", data=tel_dict[tel_type]['telescopeAltitude'], maxshape=(None,),dtype=np.float32)
            grp["telescopeAltitude"].attrs["units"] = "rad"
            grp.create_dataset("telescopeAzimuth", data=tel_dict[tel_type]['telescopeAzimuth'], maxshape=(None,), dtype=np.float32)
            grp["telescopeAzimuth"].attrs["units"] = "rad"
            grp.create_dataset("pixelsPosition", data=tel_dict[tel_type]['pixelsPosition'], dtype=np.float32)
            grp.create_dataset("injTable", data=tel_dict[tel_type]['injTable'], dtype=np.float32)
            grp.attrs["nbRow"]=tel_dict[tel_type]['nbRow']
            grp.attrs["nbCol"]=tel_dict[tel_type]['nbCol']
        # Shower simulation data
        shower_simu_grp = hdf5_file.create_group("showerSimu")
        shower_data_altitude = shower_simu_grp.create_dataset('altitude',data=altitude,maxshape=(None,),dtype=np.float32)
        shower_data_altitude.attrs["units"] = "rad"
        shower_data_azimuth = shower_simu_grp.create_dataset('azimuth',data=azimuth,maxshape=(None,),dtype=np.float32)
        shower_data_azimuth.attrs["units"] = "rad"
        shower_simu_grp.create_dataset('cmax',data=cmax,maxshape=(None,),dtype=np.float32)
        shower_simu_grp.create_dataset('depthStart',data=depthStart,maxshape=(None,),dtype=np.float32)
        shower_simu_grp.create_dataset('emax',data=emax,maxshape=(None,),dtype=np.float32)
        shower_data_energy = shower_simu_grp.create_dataset('energy',data=energy,maxshape=(None,),dtype=np.float32)
        shower_data_energy.attrs["units"] = "TeV"
        shower_data_height = shower_simu_grp.create_dataset('heightFirstInteraction',data=heightFirstInteraction,maxshape=(None,),dtype=np.float32)
        shower_data_height.attrs["units"] = "m"
        shower_simu_grp.create_dataset('hmax',data=hmax,maxshape=(None,),dtype=np.float32)
        shower_simu_grp.create_dataset('showerId',data=showerId,maxshape=(None,),dtype=np.uint64)
        shower_simu_grp.create_dataset('runId', data=run_id_shower, maxshape=(None,), dtype=np.uint64)
        shower_simu_grp.create_dataset('xmax',data=xmax,maxshape=(None,),dtype=np.float32)
        # Event simulation data
        event_simu_grp = hdf5_file.create_group("eventSimu")
        event_simu_grp.create_dataset('eventId',data=event_id_sim,maxshape=(None,),dtype=np.uint64)
        event_simu_grp.create_dataset('showerId',data=shower_id_sim,maxshape=(None,),dtype=np.uint64)
        event_simu_grp.create_dataset('runId', data=run_id_event, maxshape=(None,), dtype=np.uint64)
        event_data_xCore = event_simu_grp.create_dataset('xCore',data=xCore,maxshape=(None,),dtype=np.float32)
        event_data_xCore.attrs["units"] = "m"
        event_data_xCore.attrs["origin"] = "center of the site"
        event_data_yCore = event_simu_grp.create_dataset('yCore',data=yCore,maxshape=(None,),dtype=np.float32)
        event_data_yCore.attrs["units"] = "m"
        event_data_yCore.attrs["origin"] = "center of the site"
        # Telescope Infos
        tel_infos = hdf5_file.create_group("telescopeInfos")
        tel_infos.attrs["run"] = run_id
        tel_infos.create_dataset('telescopeId',data=tel_id,dtype=np.uint64)
        tel_data_position = tel_infos.create_dataset('telescopePosition',data=tel_position,dtype=np.float32)
        tel_data_position.attrs["units"] = "m"
        tel_data_position.attrs["origin"] = "center of the site"
        tel_infos.create_dataset('telescopeFocal',data=tel_focal,dtype=np.float32)

    hdf5_file.close()


def squeeze_data(l):
    """
    Squeeze an array or a list or a tuple of length 1
    Parameters
    ----------
    l: np.ndarray or list or tuple

    Returns
    -------
    variable
    """
    while isinstance(l,(np.ndarray, list, tuple)):
        if len(l) == 1:
            l = l[0]
        else:
            break

    return l

def extract_random_image_data_from_hdf5(hdf5filename):
    """
    Extract all the data linked to a random image from a random type of telescope
    Parameters
    ----------
    hdf5filename: string

    Returns
    -------
    dictionary
    """
    with h5py.File(hdf5filename,'r') as f:
        event_data = {}
        cam_num = np.random.choice(len(f['Cameras'].keys()))
        cam_type = list(f['Cameras'].keys())[cam_num]
        cam = 'Cameras/' + cam_type
        event_index = np.random.choice(len(f[cam+'/eventId']))
        run_id = f[cam+'/runId'][event_index]
        event_data['pcalibrun'] = f['/pcalibrun_files'].attrs[str(run_id)]
        event_data['psimu'] = f['psimu_files'].attrs[str(run_id)]
        event_data['telescope_type'] = list(_TELESCOPE_TYPE_DICT.keys())[list(_TELESCOPE_TYPE_DICT.values()).index(cam_type)]
        event_data['eventId'] = f[cam+'/eventId'][event_index]
        event_data['image'] = f[cam+'/images'][event_index]
        event_data['telescopeAltitude'] = f[cam + '/telescopeAltitude'][event_index]
        event_data['telescopeAzimuth'] = f[cam + '/telescopeAzimuth'][event_index]
        event_data['telescopeId'] = f[cam + '/telescopeId'][event_index]
        event_data['pixelsPosition'] = np.squeeze(f[cam + '/pixelsPosition'])
        event_simu_index =  squeeze_data(np.where((np.array(f['eventSimu/runId']) == run_id) & (np.array(f['eventSimu/eventId']) == event_data['eventId'])))
        event_data['showerId'] = f['eventSimu/showerId'][event_simu_index]
        event_data['xCore'] = f['eventSimu/xCore'][event_simu_index]
        event_data['yCore'] = f['eventSimu/yCore'][event_simu_index]
        shower_index = squeeze_data(np.where((np.array(f['showerSimu/runId']) == run_id) & (np.array(f['showerSimu/showerId']) == event_data['showerId'])))
        event_data['hmax'] = f['showerSimu/hmax'][shower_index]
        event_data['xmax'] = f['showerSimu/xmax'][shower_index]
        event_data['cmax'] = f['showerSimu/cmax'][shower_index]
        event_data['azimuth'] = f['showerSimu/azimuth'][shower_index]
        event_data['altitude'] = f['showerSimu/altitude'][shower_index]
        event_data['energy'] = f['showerSimu/energy'][shower_index]
        event_data['heightFirstInteraction'] = f['showerSimu/heightFirstInteraction'][shower_index]
        event_data['emax'] = f['showerSimu/emax'][shower_index]
        event_data['depthStart'] = f['showerSimu/depthStart'][shower_index]
        event_data['telescopePosition'] = f['telescopeInfos/telescopePosition'][np.array(f['telescopeInfos/telescopeId']) == event_data['telescopeId'],:]
        event_data['telescopeFocal'] = f['telescopeInfos/telescopeFocal'][np.array(f['telescopeInfos/telescopeId']) == event_data['telescopeId']]

        for key in list(event_data.keys()):
            event_data[key]=squeeze_data(event_data[key])
        return event_data

def extract_image_data_from_pcalibrun(data_extracted_from_hdf5,path_to_files):
    """
    Extract from origin pcalibrun and psim files the data related to those extracted from hdf5
    Then compare them
    Parameters
    ----------
    data_extracted_from_hdf5: dictionary
    path_to_files: string

    Returns
    -------

    """
    # Extract data
    event_data = {}
    event_data['pcalibrun'] = data_extracted_from_hdf5['pcalibrun']
    event_data['psimu'] = data_extracted_from_hdf5['psimu']

    pr, ps = load_cta_data(path_to_files + event_data['pcalibrun'],path_to_files + event_data['psimu'])

    event_data['eventId'] = data_extracted_from_hdf5['eventId']
    event_data['showerId'] = data_extracted_from_hdf5['showerId']
    event_data['telescopeId'] = data_extracted_from_hdf5['telescopeId']
    event_data['image'] = [ev.tabPixel for tel in pr.tabTelescope if tel.telescopeId == event_data['telescopeId'] for ev in tel.tabTelEvent if ev.eventId == event_data['eventId']]
    event_data['pixelsPosition'] = np.squeeze([tel.tabPos.tabPixelPosXY for tel in pr.tabTelescope if tel.telescopeId == event_data['telescopeId']])
    event_data['telescope_type'] = [tel.telescopeType for tel in pr.tabTelescope if tel.telescopeId == event_data['telescopeId']]
    event_data['telescopeAltitude'] = pr.header.altitude
    event_data['telescopeAzimuth'] = pr.header.azimuth
    event_data['xCore'] = [ev.xCore for ev in ps.tabSimuEvent if (ev.id == event_data['eventId']) & (ev.showerNum == event_data['showerId'])]
    event_data['yCore'] = [ev.yCore for ev in ps.tabSimuEvent if (ev.id == event_data['eventId']) & (ev.showerNum == event_data['showerId'])]
    event_data['hmax'] = [sh.hmax for sh in ps.tabSimuShower if sh.id == event_data['showerId']]
    event_data['xmax'] = [sh.xmax for sh in ps.tabSimuShower if sh.id == event_data['showerId']]
    event_data['cmax'] = [sh.cmax for sh in ps.tabSimuShower if sh.id == event_data['showerId']]
    event_data['azimuth'] = [sh.azimuth for sh in ps.tabSimuShower if sh.id == event_data['showerId']]
    event_data['altitude'] = [sh.altitude for sh in ps.tabSimuShower if sh.id == event_data['showerId']]
    event_data['energy'] = [sh.energy for sh in ps.tabSimuShower if sh.id == event_data['showerId']]
    event_data['heightFirstInteraction'] = [sh.heightFirstInteraction for sh in ps.tabSimuShower if sh.id == event_data['showerId']]
    event_data['emax'] = [sh.emax for sh in ps.tabSimuShower if sh.id == event_data['showerId']]
    event_data['depthStart'] = [sh.depthStart for sh in ps.tabSimuShower if sh.id == event_data['showerId']]
    tel_id = np.array([tel.telescopeId for tel in pr.tabTelescope])
    tel_position = np.array([posTel for posTel in pr.header.tabPosTel])
    tel_focal = np.array([focalTel for focalTel in pr.header.tabFocalTel])
    event_data['telescopePosition'] = np.squeeze(tel_position[np.where(tel_id == event_data['telescopeId'])])
    event_data['telescopeFocal'] = tel_focal[np.where(tel_id == event_data['telescopeId'])]

    for key in list(event_data.keys()):
        event_data[key] = squeeze_data(event_data[key])

    # Compare data
    for key in list(data_extracted_from_hdf5.keys()):
        if isinstance(event_data[key],np.ndarray):
            res = np.array_equal(event_data[key] , data_extracted_from_hdf5[key])
        else:
            res = event_data[key] == data_extracted_from_hdf5[key]
        if res:
            print(key + ' matches')
        else:
            print(event_data[key])
            print(data_extracted_from_hdf5[key])
            print(key + ' doesn\'t match, verification failed !')
            break


def browse_folder(data_folder):
    """
    Browse folder given to find pcalibrun and psimu files
    Parameters
    ----------
    data_folder:string

    Returns
    -------
    set of pcalibrun files
    """
    pcalibrun_set = set()
    psimu_set = set()
    for dirname, dirnames, filenames in os.walk(data_folder):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            if ext == ".pcalibrun":
                pcalibrun_set.add(dirname + filename)
            elif ext == ".psimu":
                psimu_set.add(dirname + filename)
    file_set = pcalibrun_set & psimu_set
    if len(file_set) != len(pcalibrun_set):
        print("! ignoring pcalibrun files (no corresponding psimu) !")
    if len(file_set) != len(psimu_set):
        print("! ignoring psimu files (no corresponding pcalibrun) !")

    return file_set
