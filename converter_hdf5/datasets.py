from hipecta.data import ctaTelescope2Matrix
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from converter_hdf5 import squeeze_data


class LSTCamDataset(Dataset):
    """LST camera simulation dataset."""

    def __init__(self, hdf5_file, transform=None):
        """
        Parameters
        ----------
            hdf5_file : hdf5 file containing the data
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.hdf5_file = hdf5_file
        self.transform = transform

    def __len__(self):
        return len(self.hdf5_file['/Cameras/LSTCAM/eventId'])

    def __getitem__(self, idx):
        image = self.hdf5_file['/Cameras/LSTCAM/images'][idx]
        run_id = self.hdf5_file['/Cameras/LSTCAM/runId'][idx]
        event_id = self.hdf5_file['/Cameras/LSTCAM/eventId'][idx]
        event_simu_index = squeeze_data(np.where((np.array(self.hdf5_file['eventSimu/runId']) == run_id) & (
        np.array(self.hdf5_file['eventSimu/eventId']) == event_id)))
        xCore = self.hdf5_file['/eventSimu/xCore'][event_simu_index]
        yCore = self.hdf5_file['/eventSimu/yCore'][event_simu_index]
        shower_id = self.hdf5_file['/eventSimu/showerId'][event_simu_index]
        idx_showerSimu = squeeze_data(np.where((np.array(self.hdf5_file['showerSimu/runId']) == run_id) & (
        np.array(self.hdf5_file['showerSimu/showerId']) == shower_id)))
        energy = self.hdf5_file['/showerSimu/energy'][idx_showerSimu]
        altitude = self.hdf5_file['/showerSimu/altitude'][idx_showerSimu]
        azimuth = self.hdf5_file['/showerSimu/azimuth'][idx_showerSimu]
        tel_id = self.hdf5_file['/Cameras/LSTCAM/telescopeId'][idx]
        tel_altitude = self.hdf5_file['/Cameras/LSTCAM/telescopeAltitude'][idx]
        tel_azimuth = self.hdf5_file['/Cameras/LSTCAM/telescopeAzimuth'][idx]
        tel_position = self.hdf5_file['/telescopeInfos/telescopePosition'][
            self.hdf5_file['/telescopeInfos/telescopeId'] == tel_id]

        telescope = np.array([tel_altitude, tel_azimuth, tel_position[0], tel_position[1]])

        labels = np.array([energy, altitude, azimuth, xCore, yCore])

        sample = {'image': image, 'telescope': telescope, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample


class LSTCamShiftedDataset(Dataset):
    """LST camera simulation dataset."""

    def __init__(self, hdf5_file, transform=None):
        """
        Parameters
        ----------
            hdf5_file : hdf5 file containing the data
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.hdf5_file = hdf5_file
        self.transform = transform

    def __len__(self):
        return len(self.hdf5_file['images'])

    def __getitem__(self, idx):
        image = self.hdf5_file['images'][idx]
        telescope = self.hdf5_file['telescopes'][idx]
        labels = self.hdf5_file['labels'][idx]

        sample = {'image': image, 'telescope': telescope, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample


class TelescopeToSquareMatrix(object):
    """Convert telescope image of vector shape to square matrix.

    Args:
        injTable (array of float32): injunction table
        nbRow (int): number of rows of the output matrix
        nbCol (int): number of cols of the output matrix

    """

    def __init__(self, injTable, nbRow, nbCol):
        assert isinstance(injTable, np.ndarray)
        assert isinstance(nbRow, np.int64)
        assert isinstance(nbCol, np.int64)
        self.injTable = injTable
        self.nbRow = nbRow
        self.nbCol = nbCol

    def __call__(self, sample):
        image, telescope, labels = sample['image'], sample['telescope'], sample['labels']

        mat = np.zeros((self.nbRow, self.nbCol))
        mat = ctaTelescope2Matrix.telescope2matrix(mat, image, self.injTable)
        mat.shape = (1,) + mat.shape

        return {'image': mat, 'telescope': telescope, 'labels': labels}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, telescope, labels = sample['image'], sample['telescope'], sample['labels']

        return {'image': torch.from_numpy(image).double(),
                'telescope': torch.from_numpy(telescope).double(),
                'labels': torch.from_numpy(labels).double()}


