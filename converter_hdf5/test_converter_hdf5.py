import converter_hdf5
import unittest
from hipecta.dataset import get


class Converter_hdf5Test(unittest.TestCase):
    """Test case utilisé pour tester les fonctions du module converter_hdf5"""

    def setUp(self):
        """Initialisation des tests."""
        self.pcalibrunfilename = get('pcalibrun file name in CTA_Analysis/share')
        self.psimufilename = get('psimu file name in CTA_Analysis/share')
        self.data = np.ones(10,3)
        self.hdf5file = h5py.File('test_converter_hdf5.hdf5',w)
        self.hdfGroup = hdf5file.create_group('test_group')
        self.hdfDataset = hdfGroup.create_dataset('test_data',data=np.ones(2,3),maxshape=(None,3))
        self.list_to_squeeze = [[1]]
        self.array_to_squeeze = np.array([[1, 2, 3]])

    def test_load_cta_data(self):
        """Test function 'converter_hdf5.load_cta_data'."""
        pr, ps = converter_hdf5.load_cta_data(self.pcalibrunfilename, self.psimufilename)
        self.assertIs(pr, core.PCalibRun)
        self.assertIs(ps, core.PSimulation)

    def test_add_data_to_dataset(self):
        """Test function 'converter_hdf5.add_data_to_dataset'."""
        res = converter_hdf5.addDataToDataset(self.hdfDataset,self.data)
        self.assertEqual(res.shape,(12,3))

    def test_cta_to_hdf5(self):
        """Test function 'converter_hdf5.cta_to_hdf5'."""
        print("Done during conversion")

    def test_squeeze_data(self):
        """Test function 'converter_hdf5.squeeze_data'."""
        l = converter_hdf5.squeeze_data(self.list_to_squeeze)
        a = converter_hdf5.squeeze_data(self.array_to_squeeze)
        self.assertEqual(l, 1)
        self.assertEqual(a.shape, (3,))

    def test_extract_random_image_data_from_hdf5(self):
        """Test function 'converter_hdf5.extract_random_image_data_from_hdf5'."""

    def test_extract_image_data_from_pcalibrun(self):
        """Test function 'converter_hdf5.extract_image_data_from_pcalibrun'."""

    def test_browse_folder(self):
        """Test function 'converter_hdf5.browse_folder'."""
