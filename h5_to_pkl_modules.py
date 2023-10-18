import numpy as np
import pandas as pd
import h5py

# Split for different modules as file structure is not identical => will be sorted out once reprocessing is done

def h5_to_pkl(file):

    h5_cov = h5py.File(file,'r')

    # MODULE 0

    if file.__contains__('datalog'):


        df_hits = pd.DataFrame(np.array(h5_cov['hits']))
        df_events = pd.DataFrame(np.array(h5_cov['events']))

        evid = [value for value, count in zip(df_events['evid'], df_events['nhit']) for _ in range(count)]
        ts = [value for value, count in zip(df_events['ts_start'], df_events['nhit']) for _ in range(count)]
        next = [value for value, count in zip(df_events['n_ext_trigs'], df_events['nhit']) for _ in range(count)]

        df_hits['next'] = next
        df_hits['evid'] = evid

        df_hits['z'] = (df_hits['ts']-ts) * 1.648e-1
        df_hits['z'] = df_hits['z'] - 304.31
        df_hits['z'] *= -1

    # MODULE 2

    elif file.__contains__('self_trigger'):


        df_hits = pd.DataFrame(np.array(h5_cov['charge/hits/data']))
        df_events = pd.DataFrame(np.array(h5_cov['charge/events/data']))

        evid = [value for value, count in zip(df_events['id'], df_events['nhit']) for _ in range(count)]
        ts = [value for value, count in zip(df_events['ts_start'], df_events['nhit']) for _ in range(count)]
        next = [value for value, count in zip(df_events['n_ext_trigs'], df_events['nhit']) for _ in range(count)]

        df_hits['next'] = next
        df_hits['evid'] = evid

        df_hits['z'] = (df_hits['ts']-ts) * 1.648e-1
        df_hits['z'] = df_hits['z'] - 304.31
        df_hits['z'] *= -1

    # MODULE 1 & 3

    else:

        df_hits = pd.DataFrame(np.array(h5_cov['charge/hits/data']))
        df_events = pd.DataFrame(np.array(h5_cov['charge/events/data']))
        z = pd.DataFrame(np.array(h5_cov['combined/hit_drift/data']))['z']

        df_hits['z'] = z

        evid = [value for value, count in zip(df_events['id'], df_events['nhit']) for _ in range(count)]
        next = [value for value, count in zip(df_events['n_ext_trigs'], df_events['nhit']) for _ in range(count)]

        df_hits['evid'] = evid
        df_hits['next'] = next
        df_hits.loc[df_hits['iogroup'] == 1, 'z'] *= -1

    # SHARED cuts on:

    # The number of external triggers > 0
    # next > 0

    # Events that have the same start time as the first hit are rejected 
    # t0_hit - t0_event != 0 for any hits in the event

    # The z coordinate should lie inside the detector between 0 & 315 mm
    # There needs to be at least 100 hits in the respective drift volume

    df_hits = df_hits[df_hits['next']>0]

    throw_list = df_hits['evid'][df_hits['z']==0]
    df_hits = df_hits[~df_hits['evid'].isin(throw_list)]

    df_hits = df_hits[(np.abs(df_hits['z']) < 315) & (df_hits['z'] > 0)]

    value_counts = df_hits['evid'].value_counts()
    df_hits = df_hits[df_hits['evid'].isin(value_counts[value_counts >= 100].index)]

    return df_hits[['px', 'py', 'iogroup', 'q', 'z', 'evid']]
