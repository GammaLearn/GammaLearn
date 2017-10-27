import converter_hdf5
import unittest

class Converter_hdf5Test(unittest.TestCase):
    """Test case utilisé pour tester les fonctions du module converter_hdf5"""

    def setUp(self):
        """Initialisation des tests."""
        self.pcalibrunfilename = ''
        self.psimufilename = ''
        self.data = np.ones(10,3)
        self.hdf5file = h5py.File('test_converter_hdf5.hdf5',w)
        self.hdfGroup = hdf5file.create_group('test_group')
        self.hdfDataset = hdfGroup.create_dataset('test_data',data=np.ones(2,3),maxshape=(None,3))

    def test_loadCTAData(selfself):
        """Test le fonctionnement de la fonction 'converter_hdf5.loadCTAData'."""



    def test_addDataToDataset(self):
        """Test le fonctionnement de la fonction 'converter_hdf5.addDataToDataset'."""
        res = converter_hdf5.addDataToDataset(self.hdfDataset,self.data)
        self.assertEqual(res.shape,(12,3))