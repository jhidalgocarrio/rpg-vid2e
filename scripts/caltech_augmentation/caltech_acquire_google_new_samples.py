import os
import re
import argparse
import glob
import cv2
from multiprocessing import Pool
from google_images_download import google_images_download   #importing the library
from functools import partial


def FLAGS():
    parser = argparse.ArgumentParser("Augment Caltech101 classes")

    parser.add_argument("--dataset", help="Path to Caltech101.", default="", required=True )
    parser.add_argument("--output", help="Path to output.", default="", required=True)
    parser.add_argument("--anumber", help="Absolute or total desired number of samples (including the new augmented samples)", required=True)
    parser.add_argument("--idx", help="Index.", default="*")
    parser.add_argument("--label", help="Label or classes name.", default="*")
    parser.add_argument("--keywords", help="Keywords to help the search."+
    "\tIf you simply type the keyword, Google will best try to match it"+
    "\tIf you want to search for exact phrase, you can wrap the keywords in double quotes ("")"+
    "\tIf you want to search to contain either of the words provided, use OR between the words.",
    default="")
    parser.add_argument("--verbose", help="Verbose mode", default=False)
    args = parser.parse_args()

    assert os.path.isdir(args.dataset), "%s should be a valid directory." % args.output
    return parser.parse_args()

def augment_class(verbose, root_dir, total_number, keywords, li):
    i, l = li
    if not os.path.isfile(l):
        _, label = os.path.split(l)

    existing_number = len(os.listdir(l))
    n_download = max(0, (int(total_number) - existing_number))
    if verbose: print ("Augment_class %d %s with %d new samples" % (i,label,n_download))
    if n_download > 0:
        arguments = {"keywords":label+" "+keywords,"limit":n_download,"print_urls":verbose,
                "output_directory": root_dir, "image_directory": label,
                "no_download":False, "chromedriver": "/usr/bin/chromedriver"}   #creating list of arguments
        response = google_images_download.googleimagesdownload()   #class instantiation
        try:
            response.download(arguments)   #passing the arguments to the function
        except:
            if verbose: print("Invalid URL Exception")

        label_dir = os.path.join(root_dir,label)
        if os.path.isdir(label_dir):
            idx =  existing_number + 1
            for filename in os.listdir(label_dir):
                imagepath = os.path.join(label_dir, filename)
                if verbose: print ("Opening: " + imagepath)
                img = cv2.imread(imagepath)
                if img is not None:
                    percentage = 300.00/float(img.shape[0])#Height is fixed to 300 in ncaltech dataset
                    if verbose: print('Original Dimensions : ',img.shape)
                    r_img = cv2.resize(img, (int(img.shape[1] * percentage), 300), interpolation=cv2.INTER_AREA)
                    if verbose: print('New Dimensions : ',r_img.shape)
                    imagename = "label_"+'{:04d}'.format(idx)+"_au.png" 
                    r_imagepath = os.path.join(label_dir, imagename)
                    cv2.imwrite(r_imagepath, r_img)
                    idx = idx + 1

                # Remove old image
                os.remove(imagepath)

if __name__ == "__main__":
    flags = FLAGS()

    #read classes from folder names
    pattern = os.path.join(flags.dataset, flags.label)
    labels = glob.glob(pattern)

    # pool of threads
    pool = Pool(8)

    # Run the function
    function=partial(augment_class, flags.verbose, flags.output, flags.anumber, flags.keywords)
    pool.map_async(function, enumerate(labels)).get(9999999)
