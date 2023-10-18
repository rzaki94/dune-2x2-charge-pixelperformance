import numpy as np
import pandas as pd
from houghUtils import get_pixels_subset
import sys

# Script that creates the two maps:
# 1: Hit pixels => 2D map with only the pixels that recorded charge [recorded == True] in getRandomSeg
# 2: All pixels => 2D map with all pixels irrespective of recorded charge

if len(sys.argv) > 5:
    file = sys.argv[1]
    module = sys.argv[2]
    subset_cat = sys.argv[3]
    inputdir = sys.argv[4]
    outputdir = sys.argv[5]

def convert_single_item_list_to_value(lst):
    if len(lst) == 1:
        return lst[0]
    else:
        return lst

def plot_2d_dist_short(file, module, save=True, subset_cat = ''):

    ps = 4.434
    rad = 16

    if file.__contains__('self_trigger'):
        ps = 3.8
        rad = 14
        module = 'mod2'

    if file.__contains__("datalog"):
        module = 'mod0'

    if file.__contains__("events"):
        module = 'mod1'

    if file.__contains__("tpc12-reco-2023"):
        module = 'mod3'

    DF = pd.read_pickle(inputdir + file)

    # Categories to separate the pixels going to the plotting function
    # Used to study the efficiency of different regions on the pixels plane
    # Cat 1: Corners
    # Cat 2: Edges top
    # Cat 3: Edges sides
    # Cat 4: Tile central regions
    # Cat 5: Tile boundaries horizontal
    # Cat 6: Tile boundaries vertical

    val = 20
    bin_2d_x = np.arange(-5 * ps / 2, 5 * ps / 2 + ps / val, ps / val)
    bin_2d_y = np.arange(-5 * ps / 2, 5 * ps / 2 + ps / val, ps / val)

    DF_TRUE = DF[DF['recorded']==True]

    all_hist, binsx, binsy = np.histogram2d(DF['d_hl_0x_res'].values.reshape(len(DF['d_hl_0x_res'])), DF['d_hl_0y_res'].values.reshape(len(DF['d_hl_0y_res'])), bins=[bin_2d_x, bin_2d_y])
    hit_hist, binsx, binsy = np.histogram2d(DF_TRUE['d_hl_0x_res'], DF_TRUE['d_hl_0y_res'], bins=[bin_2d_x, bin_2d_y])

    if save == True:
        np.save(outputdir + module + "/all_pixels/all_pixels_" + file.split('/')[-1] + ".npy", all_hist)
        np.save(outputdir + module + "/hit_pixels/hit_pixels_" + file.split('/')[-1] + ".npy", hit_hist)

    return all_hist, hit_hist
