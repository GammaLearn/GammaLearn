import converter_hdf5 as cv
import argparse
import os

# Example of use. Need to define the way to get the file names
# pruncalibfilename = '/home/jacquemont/projets_CTA/Prod3b/Paranal/Gamma_point_source/gamma_20deg_0deg_run4514___cta-prod3_desert-2150m-Paranal-merged.pcalibRun'
# simufilename = '/home/jacquemont/projets_CTA/Prod3b/Paranal/Gamma_point_source/gamma_20deg_0deg_run4514___cta-prod3_desert-2150m-Paranal-merged.psimu'
# hdf5filename = '/home/jacquemont/projets_CTA/gamma.hdf5'

parser = argparse.ArgumentParser()
# parser.add_argument("pcalibrunfile", help="pcalibrun file name to read")
# parser.add_argument("psimulation", help="psimulation file name to read")
parser.add_argument("data_folder", help="path to folder of pcalibrun and psimu files")
parser.add_argument("hdf5file", help="hdf5 file name (including path) to write")
args = parser.parse_args()
#pruncalibfilename = args.pcalibrunfile
#simufilename = args.psimulation
data_folder = args.data_folder + '/'
hdf5filename = args.hdf5file


#cta_to_hdf5(pruncalibfilename, simufilename, hdf5filename)
# dic=extract_random_image_data_from_hdf5(hdf5filename)
# path_to_files = '/home/jacquemont/projets_CTA/Prod3b/Paranal/Gamma_point_source/'
# extract_image_data_from_pcalibrun(dic,path_to_files)

prset = cv.browse_folder(data_folder)
prlist = list(prset)
prlist.sort()
chunk_size = 10
pr_chunks = [prlist[i:i + chunk_size] for i in range(0, len(prlist), chunk_size)]
for i, chunk in enumerate(pr_chunks):
    print("Process chunk %d over %d" % (i + 1, len(pr_chunks)))
    hdf5name, ext = os.path.splitext(hdf5filename)
    hdf5name += str(i)
    hdf5name += ext
    for file in chunk:
        print("Convert file : ", file)
        cv.cta_to_hdf5(file + ".pcalibrun", file + ".psimu", hdf5name)
    # Random checking of data in hdf5 file
    print("Check conversion")
    print("Extract random image data from hdf5 file : ", hdf5name)
    dic = cv.extract_random_image_data_from_hdf5(hdf5name)
    print("Extract related data in pcalibrun and psimu files")
    cv.extract_image_data_from_pcalibrun(dic, data_folder)