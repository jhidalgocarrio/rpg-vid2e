import os
import glob
import argparse

def FLAGS():


root = ""
root = "../mpi_inf_3dhp/"

videos = glob.glob(root + "S*/*/imageSequence/*.avi")


print(root + "S*/*/*.avi", videos)
for vid in videos:
    folder = vid[:-4]
    if not os.path.exists(folder):
        os.mkdir(folder)
    os.system("ffmpeg -i %s %s/image_%%05d.png" % (vid, folder))
