import numpy as np
import pickle
from houghUtils import EnergyInRadiusLine, GetLinesEvent, hl_endpoints, track_entry, Get_ratio_MIP
from event_to_pkl import h5_to_pkl
import warnings
warnings.filterwarnings("ignore")

def GetLinesEvents(events, filename, output_dir):

    DFHough = []

    if filename.__contains__('tpc12'):
        radius = 9
    else:
        radius = 7

    for data in events:

        if np.size(data) == 1:
            continue

        data = data[data['iogroup'] == 1]
        data['ts'][data['iogroup'] == 1]  = data['ts'][data['iogroup'] == 1] * 1.648e-1

        data = data[data['ts'] < 305]
        data = data[data['ts'] > 0]

        if len(data) < 50:
            continue

        hc_l, hits_on_hline_hcl2 = GetLinesEvent(data, radius)

        if np.size(hc_l) == 1:
            continue

        if (hits_on_hline_hcl2['start_X'] == 0.0).all():
            continue

        hc_l = hl_endpoints(hc_l)

        if (track_entry(hc_l).count(True)) != 2:
            continue

        if Get_ratio_MIP(hits_on_hline_hcl2, radius, 25) < 0.9:
            continue

        if (EnergyInRadiusLine(hits_on_hline_hcl2, 0, 8)/(10*hc_l['trackL'].values))[0] > 3:
            continue

        DFHough.append([hc_l, hits_on_hline_hcl2])

    with open(output_dir +'/DFHoughnolim' + filename.split('/')[-1][:-2] + 'pkl', 'wb') as file:
        pickle.dump(DFHough, file)

file = sys.argv[1]
output_dir = sys.argv[2]

# Calls the conversion script, but only if the hough output file does not exist

if os.path.isfile(output_dir + '/DFHoughnolim' + file[:-2] + 'pkl'):
    sys.exit()

data_io1 = h5_to_pkl('input_files_dest/' + file)[0]
print("Finished the transformation to pkl")

print("Started looking for hough lines")
GetLinesEvents(data_io1, file, output_dir)
print("Please find the output in " + output_dir)
