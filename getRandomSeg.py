import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")
import random
import sys
import numpy as np
import pickle 
import pandas as pd

# The segment finder script that allows us to find random point along the track separated by a 5x5 pixel array 
# Calculates the distance in X & Y of the segment to each surrounding pixel

if len(sys.argv) > 2:

    filename = sys.argv[1]
    module = sys.argv[2]
    inputdir = sys.argv[3]
    outputdir = sys.argv[4]

if filename.__contains__("_io_1_"):
    iogroup = "1"
if filename.__contains__("_io_2_"):
    iogroup = "2"

# Function to determine if consecutive random points on the line are within a 5x5 pixel array from one another => REJECT

def remove_close_values(values, y):
    values.sort()  # Sort the values in ascending order
    i = 0
    while i < len(values) - 1:
        if abs(values[i] - values[i + 1]) < y:
            # Remove one of the close values
            del values[i + 1]
        else:
            i += 1


def findpixelefficiency(filename, module):

    with open(inputdir + filename, 'rb') as file:
        DFHough = pickle.load(file)

    # distance is calculated from random point to all pixels on the readout plane
    # lim is the limit beyond which pixels are not considered

    # radius is the length of the segment we want to find the random point on
    # currently set to facilitate a 7x7 array

    radius = 44
    lim = 3.5*4.434

    # load the full working pixel map (total map - broken pixels)

    if module == 'mod0':
        df_pixel_map = pd.read_pickle('./geometry/working_map_'+module+'_' + iogroup + '.pkl')

    if module == 'mod1':
        df_pixel_map = pd.read_pickle('./geometry/working_map_'+module+'_' + iogroup + '.pkl')

    if module == 'mod2':
        df_pixel_map = pd.read_pickle('./geometry/working_map_'+module+'_' + iogroup + '.pkl')

        # MODULE 2 has a 3.8 mm pixel size, change in radius and limit

        radius = 38
        lim = 3.5*3.8

    if module == 'mod3':
        df_pixel_map = pd.read_pickle('./geometry/working_map_'+module+'_' + iogroup + '.pkl')

    # Facilitate the option for the full 2x2
    if module == 'mod4':
       df_pixel_map = pd.read_pickle('./geometry/2x2_pixel_map.pkl')
       radius = 50

    # Reducing storage size and ensuring we can match between full pixel map and pixel hits in the event
    df_pixel_map = df_pixel_map.astype('float32')
    df_pixel_map['start_Y'] = np.round(df_pixel_map['start_Y'], 3)
    df_pixel_map['start_X'] = np.round(df_pixel_map['start_X'], 3)

    dists = pd.DataFrame()

    for lineset, hits_on_hline_hcl2 in DFHough:

        hits_on_hline_hcl2 = hits_on_hline_hcl2.astype('float32')
        hits_on_hline_hcl2['start_Y'] = np.round(hits_on_hline_hcl2['start_Y'], 3)
        hits_on_hline_hcl2['start_X'] = np.round(hits_on_hline_hcl2['start_X'], 3)

        # finding the number of segments along the track
        line_start = -lineset['trackL'].values[0]/2
        num_segments = int(np.floor(lineset['trackL'].values[0]/radius))
        segment_length = radius

        # finding a random point in each of the segments
        random_points = [random.uniform(line_start + i * segment_length, line_start + (i + 1) * segment_length) for i in
                         range(num_segments)]

        # checking the distance between the points
        remove_close_values(random_points, radius)

        # finding the X & Y position for each random point
        ArX = lineset['aX'].values+lineset['bX'].values*random_points
        ArY = lineset['aY'].values+lineset['bY'].values*random_points

        for p in np.c_[ArX, ArY]:

            # throw out all the points on the full map that are outside the limits
            df_seg = df_pixel_map[(np.abs(p[0] - df_pixel_map['start_X'])<lim) & (np.abs(p[1] - df_pixel_map['start_Y'])<lim)]

            # calculate distance from point to pixels
            df_seg['d_hl_0x_res'] = p[0] - df_pixel_map['start_X']
            df_seg['d_hl_0y_res'] = p[1] - df_pixel_map['start_Y']

            # see if the pixel to which the distance was calculated collected any charge
            df_seg['recorded'] = ((df_seg['start_Y'].isin(hits_on_hline_hcl2['start_Y'])) & (df_seg['start_X'].isin(hits_on_hline_hcl2['start_X'])))

            dists = pd.concat([dists, df_seg])

    dists.reset_index(inplace=True, drop=True)
    dists.to_pickle(outputdir + filename[:-4]+'_pixels.pkl')

findpixelefficiency(filename, module)
