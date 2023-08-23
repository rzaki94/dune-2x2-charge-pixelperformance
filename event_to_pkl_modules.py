import numpy as np
import pandas as pd
import h5py

# The script reads in the h5 file from the individual module data runs and 
# selects only the events with a minimum of 100 hits. Then it puts out two
# files containing the hit information in the separate drift volumes

def h5_to_pkl(file):

    h5_cov = h5py.File(file,'r')

    if file.__contains__('datalog'):
        df_hits = pd.DataFrame(np.array(h5_cov['hits']))
        df_events = pd.DataFrame(np.array(h5_cov['events']))
        id = 'evid'
    else:
        df_hits = pd.DataFrame(np.array(h5_cov['charge/hits/data']))
        df_events = pd.DataFrame(np.array(h5_cov['charge/events/data']))
        id = 'id'

    hits = 0
    hit = 0

    wanted_list_io1 = []
    wanted_list_io2 = []

    for i in range(len(df_events)):
        hit = df_events['nhit'][df_events[id]==i].values[0] + hit
        df_wanted = df_hits[hits:hit][['px', 'py', 'ts', 'q', 'iogroup']]
        df_wanted['ts'] = df_wanted['ts'] - df_events['ts_start'][df_events[id]==i].values[0]

        wanted_io1 = 0
        wanted_io2 = 0

        if hit-hits < 100:
            hits = hit
            continue

        if len(df_wanted[df_wanted['iogroup'] == 1]) > 100:
            wanted_io1 = df_wanted[df_wanted['iogroup'] == 1]

        if len(df_wanted[df_wanted['iogroup'] == 2]) > 100:
            wanted_io2 = df_wanted[df_wanted['iogroup'] == 2]

        hits = hit
        wanted_list_io1.append(wanted_io1)
        wanted_list_io2.append(wanted_io2)

    return wanted_list_io1, wanted_list_io2


