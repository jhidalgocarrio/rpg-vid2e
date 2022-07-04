"""python file for converting folder of rosbags to n-caltech101 classification format.
The rosbag should have the following structure:
- root
    - class1
        image_0001.bin
        :
        :
    - class2
        image_0001.bin
    ...

The rosbags should have the following topics:
    /dvs/events: events, in dvs_EventArray msgs
    /dvs/image_raw: image message
    /dvs_rendering: rendered images with events and frames


The output has the following format:

- output
    - class1
        events_0001.pkl
        images_0001.pkl
        renderings_0001.pkl
        events_0002.pkl
        images_0002.pkl
        renderings_0002.pkl
        :
    - class2
        :
    :

"""

from multiprocessing import Pool
import argparse
import os
import sys
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "AedatTools"))
from AedatTools.PyAedatTools.ImportAedatHeaders import ImportAedatHeaders


def ImportBin(aedat):
    # unpack
    info = aedat['info']
    importParams = aedat['importParams']
    fileHandle = importParams['fileHandle']

    # The formatVersion dictates whether there are 6 or 8 bytes per event.
    numBytesPerEvent = 5
    addrPrecision = np.dtype([('x', '>u1'), ('yts', '>u4')])

    # Find the number of events, assuming that the file position is just at the
    # end of the headers.
    fileHandle.seek(0, 2)
    info['numEventsInFile'] = int(np.floor(
        (fileHandle.tell() - info['beginningOfDataPointer']) /
        numBytesPerEvent))

    # Check the startEvent and endEvent parameters
    if 'startEvent' in importParams:
        startEvent = importParams['startEvent']
    else:
        startEvent = 0
    assert startEvent <= info['numEventsInFile']
    if 'endEvent' in importParams:
        endEvent = importParams['endEvent']
    else:
        endEvent = info['numEventsInFile']
    assert endEvent <= info['numEventsInFile']
    assert startEvent <= endEvent

    numEventsToRead = endEvent - startEvent + 1

    # Read events
    fileHandle.seek(info['beginningOfDataPointer'] + numBytesPerEvent *
                    startEvent)
    allEvents = np.fromfile(fileHandle, addrPrecision, numEventsToRead)

    x = np.array(allEvents['x'])
    allTs = np.array(allEvents['yts'])

    # Interpret the addresses

    """
    Split between DVS/DAVIS and DAS.
        For DAS1:
            - Special events - external injected events has never been
            implemented for DAS
            - Split between Address events and ADC samples
            - Intepret address events
            - Interpret ADC samples
        For DVS128:
            - Special events - external injected events are on bit 15 = 1 
            there is a more general label for special events which is bit 31 =
            1, but this has ambiguous interpretations  it is also overloaded
            for the stereo pair encoding - ignore this. 
            - Intepret address events
        For DAVIS:
            - Special events
                - Interpret IMU events from special events
            - Interpret DVS events according to chip class
            - Interpret APS events according to chip class
    """

    """
        # DAVIS. In the 32-bit address:
        # bit 32 (1-based) being 1 indicates an APS sample
        # bit 11 (1-based) being 1 indicates a special event
        # bits 11 and 32 (1-based) both being zero signals a polarity event
    """

    # Create a structure to put all the data in
    outputData = {}

    yMask = int('11111111000000000000000000000000', 2)
    yShiftBits = 24
    pMask = int('00000000100000000000000000000000', 2)
    pShiftBits = 23
    tMask = int('00000000011111111111111111111111', 2)

    # Polarity(DVS) events
    outputData["polarity"] = {}
    outputData["polarity"]['x'] = x
    outputData["polarity"]['y'] = np.array(np.right_shift(
        np.bitwise_and(allTs, yMask), yShiftBits), 'uint8')
    outputData["polarity"]['timeStamp'] = np.array(
        np.bitwise_and(allTs, tMask), 'uint32')
    outputData["polarity"]['polarity'] = np.array(
        np.right_shift(np.bitwise_and(allTs, pMask), pShiftBits), 'bool')
    # Polarity bit
    del allTs

    aedat['info'] = info
    aedat['data'] = outputData

    return aedat


def bin_to_numpy(f):
    aedat = {}
    aedat['importParams'] = {}
    aedat['importParams']['filePath'] = f

    with open(aedat['importParams']['filePath'], 'rb') as aedat['importParams']['fileHandle']:
        aedat = ImportAedatHeaders(aedat, comment_character="sdfjalkd")
        aedat = ImportBin(aedat)

    data = aedat["data"]["polarity"]
    x, y = data["x"], data["y"]
    polarity = 2*data["polarity"].astype(np.float32)-1
    ts = data["timeStamp"].astype(np.float32)/1e6

    array = np.stack([x, y, ts, polarity], 1)

    return array

def FLAGS():
    parser = argparse.ArgumentParser("Convert Bag folder into N-Caltech folder")
    # root and output folder
    parser.add_argument("--dataset_root", help="Root where the rosbags are.", default="/home/dani/code/catkin_ws/src/rpg_vid2e/scripts/rosbags")
    parser.add_argument("--output_root", help="Root where the output should be.", default="/home/dani/code/catkin_ws/src/rpg_vid2e/scripts/rosbags")

    parser.add_argument("--num_workers", help="Number of threads for converting.", type=int, default=4)

    args = parser.parse_args()

    assert os.path.isdir(args.dataset_root), "%s should be valid dir." % args.dataset_root
    assert os.path.isdir(args.output_root), "%s should be a valid dir." % args.output_root

    return parser.parse_args()


def parse_file(f):
    """File has shape class/image_0001.bin"""
    label = os.path.basename(os.path.dirname(f))
    counter = f[-8:-4]
    return label, counter

def load_data_from_bin(file):
    data = bin_to_numpy(file)
    x, y, t, polarity = data.T
    dataframe = pd.DataFrame({
        "x": x.astype(np.uint8),
        "y": y.astype(np.uint8),
        "time": t.astype(np.float32),
        "polarity": (1==polarity).astype(np.bool)
    })
    return dataframe

def convert_1_file(idx_and_file):
    i, file = idx_and_file
    label, counter = parse_file(file)
    label_root = os.path.join(dataset_root, label)

    if not os.path.isdir(label_root):
        os.mkdir(label_root)

    print("[%4d/%4d] - Converting file %s -> %s" % (
        i, len(files), file, os.path.join(label_root, "cam0_events_%s.pkl" % counter)))

    dataframe = load_data_from_bin(file)
    dataframe.to_pickle(os.path.join(label_root, "cam0_events_%s.pkl" % counter))


if __name__ == '__main__':
    flags = FLAGS()

    pattern = os.path.join(flags.dataset_root, "*", "*.bin")
    files = glob.glob(pattern)

    # create output folder
    dataset_root = os.path.join(flags.output_root, "dataset")
    print("Making dataset root at %s." % (dataset_root))
    if not os.path.isdir(dataset_root):
        os.mkdir(dataset_root)

    pool = Pool(flags.num_workers)
    pool.map_async(convert_1_file, enumerate(sorted(files))).get(999999999)  # workaround since otherwise cannot be cancelled