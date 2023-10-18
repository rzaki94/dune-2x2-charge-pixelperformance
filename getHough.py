import sys
import numpy as np
import pickle 
from houghUtils import EnergyInRadiusLine, GetLinesEvent, hl_endpoints, track_entry, Get_ratio_MIP
import pandas as pd
from h5_to_pkl_modules import h5_to_pkl
import warnings
warnings.filterwarnings("ignore")

# Hough transform algorithm script
# Transforms h5 to a pkl file containing all events with ids
# Finds hough lines for each event & puts out a list of line, hits

# Define the filename, directory & desired drift volume
if len(sys.argv) > 4:

    filename = sys.argv[1]
    inputdir = sys.argv[2]
    io = int(sys.argv[3])
    outputdir = sys.argv[4]

# Set chunk_size for #events to split the file at
# Module 2 & 3 generally have fewer events per file (*tpc12*)

chunk_size = 12500
radius = 1000

if filename.__contains__('tpc12'):
    chunk_size = 2500

def GetLinesEvents(events, filename, radius, io, chunk_id):

    # Minimum number of pixels to classify as a line
    # Maximum number of allowed lines per events

    min_pixels = 100
    MAX_peaks = 5

    DFHough = []
    DFevents = []

    # Require module 2 to have at least 250 hits per desired drift volume as opposed to 100 for the other three

    if filename.__contains__("self_trigger"):
        value_counts = events['evid'][events['iogroup']==int(io)].value_counts()
        events = events[events['evid'].isin(value_counts[value_counts >= 100].index)]

    # Single file with all events => use ID to separate events

    for id in events['evid'].unique():

        data = events[events['evid']==id]
        data = data[data['iogroup'] == int(io)]

        # Dump empty dataframes (artifact)

        if np.size(data) == 1:
            print('jaja')
            continue

        # Dump dataframe with fewer than 100 points (artifact)

        if len(data) < 100:
            print('jajaja')
            continue

        # Function that performs the HTA

        try:
            hc_l, hits_on_hline_hcl2 = GetLinesEvent(data, filename, min_pixels, MAX_peaks, radius)

        except:
            continue

        # Dump dataframe that has only 0's for the x values (artifact)

        if (hits_on_hline_hcl2['start_X'] == 0.0).all():
            continue

        # Find track start/end points

        hc_l = hl_endpoints(hc_l)

        # Ensure track goes through two surfaces

        if (track_entry(hc_l, plot=False).count(True)) != 2:
            continue

        # Ensure 90% of energy is contained within first 10 mm as opposed to first 25 mm

        if Get_ratio_MIP(hits_on_hline_hcl2, 10, 25) < 0.9:
            continue

        # Ensure collected charge is smaller than 3/mm for each individual track

        if (EnergyInRadiusLine(hits_on_hline_hcl2, 0, 8)/(10*hc_l['trackL'].values))[0] > 3:
            continue

        DFHough.append([hc_l, hits_on_hline_hcl2])

    with open(outputdir + '/' +  filename.split('/')[-1][:-3] + '_io_' + str(io) + '_chunk_' + str(chunk_id) + '.pkl', 'wb') as file:
        pickle.dump(DFHough, file)

data_io = h5_to_pkl(inputdir + filename)

print("Finished the transformation to pkl")

unique_evid = data_io['evid'].unique()

for id, i in enumerate(range(0, len(unique_evid), chunk_size)):

    chunk_evid = unique_evid[i:i + chunk_size]
    chunk = data_io[data_io['evid'].isin(chunk_evid)]
    GetLinesEvents(chunk, filename, radius, io, id)
