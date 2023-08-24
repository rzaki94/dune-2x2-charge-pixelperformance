import pandas as pd
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from skspatial.objects import Plane
from skspatial.objects import Line

# Utility functions that support the main script

# First sections contains the main function in GetLinesEvent, which is called from "get_hough" to find hough lines in each individual event

def GetLinesEvent(event, radius):

    min_pixels = 100
    MAX_peaks = 5

    if type(event) == int:
        return 0, 0

    event.rename(columns={'px': 'start_X', 'py': 'start_Y', 'ts': 'start_Z', 'q': 'EnergyDeposit'}, inplace=True)

    df_hough_w_endpoints = HoughOnArray(event, min_pixels, MAX_peaks)

    if len(df_hough_w_endpoints) != 1:
        return 0, 0

    d_pnt2line(event, df_hough_w_endpoints)
    hc_l = get_hlcenter()

    if np.abs(hc_l['bZ'].values) < 5 * 10e-4:
        hc_l['npoints'] = 0
        return 0, 0

    try:
        hits_on_hline_hcl2, track_L_hcl2 = trackLengthhoughcenter(event, hc_l.iloc[0], 0, 10)

    except:
        hc_l['npoints'] = 0
        return 0,0

    hits_on_hline_hcl = hits_on_hline_hcl2[hits_on_hline_hcl2['d_hl_0']<radius]
    track_L_hcl = hits_on_hline_hcl['d_2_avg2'].nlargest(1).values

    hits_on_hline_hcl2['start_X'] = np.round(hits_on_hline_hcl2['start_X'], decimals=3)
    hits_on_hline_hcl2['start_Y'] = np.round(hits_on_hline_hcl2['start_Y'], decimals=3)
    hits_on_hline_hcl2['start_Z'] = np.round(hits_on_hline_hcl2['start_Z'], decimals=3)

    try:
        line2 = set_newhlcenter(hits_on_hline_hcl, hc_l.iloc[0])

    except:
        hc_l['npoints'] = 0
        return 0,0

    if line2['aX'] == 0 and line2['bX'] == 0:
        line2['npoints'] = 0
        return 0,0

    line2['npoints'] = track_L_hcl[0]
    hc_l.rename(columns={'npoints': 'trackL'}, inplace=True)
    hc_l.iloc[0] = line2
    return hc_l, hits_on_hline_hcl2



def HoughOnArray(arrayname, min_pixels=125, MAX_peaks=100):
    df_LAr = arrayname

    df_LAr.to_csv('./hough3D_inputfile_start.csv',
                  columns=['start_X', 'start_Y', 'start_Z'], index=False, header=False)

    subprocess.check_call([r"./Hough3Dpackage/hough3dlines",
                           "./hough3D_inputfile_start.csv",
                           "-o", "./hough3D_outputfile.csv",
                           "-minvotes", str(min_pixels),
                           "-nlines", str(MAX_peaks),
                           "-raw"])

    with open('./hough3D_outputfile.csv', 'r') as fin:
        data = fin.read().splitlines(True)
        t_bounds = data[1:2][0].split()
        t_min = int(np.floor(float(t_bounds[0])))
        t_max = int(np.ceil(float(t_bounds[1])))
        t_space = np.array([t_min, t_max])

    with open('./hough3D_outputfile.csv', 'w') as fout:
        fout.writelines(data[2:])

    df_hough = pd.read_csv("./hough3D_outputfile.csv",
                           delim_whitespace=True, names=['aX', 'aY', 'aZ', 'bX', 'bY', 'bZ', 'npoints'])

    df_hough['xh_i'] = df_hough['aX'].values + t_space[0] * df_hough['bX'].values
    df_hough['xh_f'] = df_hough['aX'].values + t_space[1] * df_hough['bX'].values
    df_hough['yh_i'] = df_hough['aY'].values + t_space[0] * df_hough['bY'].values
    df_hough['yh_f'] = df_hough['aY'].values + t_space[1] * df_hough['bY'].values
    df_hough['zh_i'] = df_hough['aZ'].values + t_space[0] * df_hough['bZ'].values
    df_hough['zh_f'] = df_hough['aZ'].values + t_space[1] * df_hough['bZ'].values

    return df_hough

def d_pnt2line(df_True, df_hough_w_endpoints):
    A = df_True[['start_X', 'start_Y', 'start_Z']]
    B = np.array([df_hough_w_endpoints['xh_i'], df_hough_w_endpoints['yh_i'], df_hough_w_endpoints['zh_i']]).T
    C = np.array([df_hough_w_endpoints['xh_f'], df_hough_w_endpoints['yh_f'], df_hough_w_endpoints['zh_f']]).T

    for idx, (b, c) in enumerate(zip(B, C)):
        d = (c - b) / np.linalg.norm(c - b)
        v = A - b
        t = np.dot(v, d)
        P = np.array([t_temp * d + b for t_temp in t])
        DIST = np.sqrt(np.einsum('ij,ij->i', P - A, P - A))
        df_True['d_hl_' + str(idx)] = DIST

    return df_True


def hl_endpoints(hc_l):
    hc_l['xh_i'] = hc_l['aX'].values + 0.5 * hc_l['trackL'] * hc_l['bX'].values
    hc_l['xh_f'] = hc_l['aX'].values - 0.5 * hc_l['trackL'] * hc_l['bX'].values
    hc_l['yh_i'] = hc_l['aY'].values + 0.5 * hc_l['trackL'] * hc_l['bY'].values
    hc_l['yh_f'] = hc_l['aY'].values - 0.5 * hc_l['trackL'] * hc_l['bY'].values
    hc_l['zh_i'] = hc_l['aZ'].values + 0.5 * hc_l['trackL'] * hc_l['bZ'].values
    hc_l['zh_f'] = hc_l['aZ'].values - 0.5 * hc_l['trackL'] * hc_l['bZ'].values

    return hc_l

def d_pnt2line_test_XY(df, df_hough_w_endpoints):

    df_True = df.copy(deep=True)

    A = df_True[['start_X', 'start_Y']]
    B = np.array([df_hough_w_endpoints['xh_i'], df_hough_w_endpoints['yh_i']]).T
    C = np.array([df_hough_w_endpoints['xh_f'], df_hough_w_endpoints['yh_f']]).T

    for idx, (b, c) in enumerate(zip(B, C)):
        d = (c - b) / np.linalg.norm(c - b)
        v = A - b
        t = np.dot(v, d)
        P = np.array([t_temp * d + b for t_temp in t])

        df_True['d_hl_' + str(idx) + 'x_res'] = (A - P)['start_X']
        df_True['d_hl_' + str(idx) + 'y_res'] = (A - P)['start_Y']

        DIST = np.sqrt(np.einsum('ij,ij->i', P - A, P - A))
        df_True['d_hl_' + str(idx)] = DIST

    return df_True


def EnergyInRadiusLine(df, line_num, R):
    df_hits_on_hline = df.copy()

    mask = df['d_hl_' + str(line_num)] < R
    df_hits_on_hline = df_hits_on_hline[mask]
    energy_T_hcl = df_hits_on_hline['EnergyDeposit'].sum()

    return energy_T_hcl

def trackLengthhoughcenter(df, hl_c, line_num, radius):
    df_hits_on_hline = df.copy()
    mask = df['d_hl_' + str(line_num)] < radius
    df_hits_on_hline = df_hits_on_hline[mask]

    # TODO think about the hit at 0.0

    avg = np.array([df_hits_on_hline['start_X'] - hl_c['aX'],
                    df_hits_on_hline['start_Y'] - hl_c['aY'],
                    df_hits_on_hline['start_Z'] - hl_c['aZ']])

    df_hits_on_hline['d_2_avg'] = np.linalg.norm(avg, axis=0)


    avg2 = np.array([df_hits_on_hline['start_X'] - df_hits_on_hline.nlargest(1, 'd_2_avg')['start_X'].values,
                     df_hits_on_hline['start_Y'] - df_hits_on_hline.nlargest(1, 'd_2_avg')['start_Y'].values,
                     df_hits_on_hline['start_Z'] - df_hits_on_hline.nlargest(1, 'd_2_avg')['start_Z'].values])
    df_hits_on_hline['d_2_avg2'] = np.linalg.norm(avg2, axis=0)

    return df_hits_on_hline, df_hits_on_hline['d_2_avg2'].nlargest(1).values


def get_hlcenter():
    hc_l = pd.read_csv("./hough3D_outputfile.csv",
                       delim_whitespace=True, names=['aX', 'aY', 'aZ', 'bX', 'bY', 'bZ', 'npoints'])

    return hc_l


def set_newhlcenter(hits_on_hline_hcl, line):

    avg = np.array([(hits_on_hline_hcl.nlargest(1, 'd_2_avg')['start_X'].values[0] - line['aX']) / line['bX'],
                    (hits_on_hline_hcl.nlargest(1, 'd_2_avg')['start_Y'].values[0] - line['aY']) / line['bY'],
                    (hits_on_hline_hcl.nlargest(1, 'd_2_avg')['start_Z'].values[0] - line['aZ']) / line['bZ']])

    if np.all((avg < 0)):
        coeff = -1
    else:
        coeff = 1

    shift = hits_on_hline_hcl.nlargest(1, 'd_2_avg')['d_2_avg'].values[0] - (
                hits_on_hline_hcl.nlargest(1, 'd_2_avg2')['d_2_avg2'].values[0] -
                hits_on_hline_hcl.nlargest(1, 'd_2_avg')['d_2_avg'].values[0])

    shift *= 0.5

    line2 = line.copy()

    line2['aX'] = line['aX'] + shift * coeff * line['bX']
    line2['aY'] = line['aY'] + shift * coeff * line['bY']
    line2['aZ'] = line['aZ'] + shift * coeff * line['bZ']

    return line2

def Get_ratio_MIP(df, R1, R2):

    E_1 = EnergyInRadiusLine(df, 0, R1)
    E_2 = EnergyInRadiusLine(df, 0, R2)

    return E_1/E_2

def point_in_vol(point):

    x0 = -315
    x1 = 315
    y0 = -615
    y1 = 615
    z0 = -300
    z1 = 300

    if x0 <= point[0] <= x1 and y0 <= point[1] <= y1 and z0 <= point[2] <= z1:
        return True

    return False

def track_entry(df):

    df_hough = df

    plane_t = Plane.from_points([300, -600, 300], [300, -600, -300], [-300, -600, 300])
    plane_b = Plane.from_points([300, 600, 300], [300, 600, -300], [-300, 600, 300])

    plane_f = Plane.from_points([300, -600, 300], [300, 600, 300], [-300, -600, 300])
    plane_r = Plane.from_points([300, -600, -300], [300, 600, -300], [-300, -600, -300])

    plane_l = Plane.from_points([-300, -600, 300], [-300, 600, 300], [-300, -600, -300])
    plane_v = Plane.from_points([300, -600, 300], [300, 600, 300], [300, -600, -300])

    for houghline in df_hough.iloc:
        line = Line([houghline['aX'], houghline['aY'], houghline['aZ']],
                    [houghline['bX'], houghline['bY'], houghline['bZ']])


        try:
            point_t = plane_t.intersect_line(line)
            point_b = plane_b.intersect_line(line)

            #front/rear
            point_f = plane_f.intersect_line(line)
            point_r = plane_r.intersect_line(line)

            #left/right
            point_l = plane_l.intersect_line(line)
            point_v = plane_v.intersect_line(line)

        except:
            continue

        point_list = [point_t, point_b, point_f, point_r, point_l, point_v]
        surv_point_list = []

        for point in point_list:
            surv_point_list.append(point_in_vol(point))

    try:
        return surv_point_list
    except:
        return [0, 0, 0]


def pixel_bins_edges():
    ps = 4.434

    bins_pixel_x_ht = np.arange(-308.173 - (ps / 2), -3, ps)
    bins_pixel_x_ht2 = -np.arange(-308.173 - (ps / 2), -3, ps)[::-1]
    bins_pixel_x = np.r_[bins_pixel_x_ht, bins_pixel_x_ht2]

    bins_pixel_y_ht = -np.arange(-303.739 - ps / 2, -3, ps)[::-1]
    bins_pixel_y_ht2 = np.arange(316.661 - ps / 2, 615, ps)
    bins_pixel_y_ht3 = np.r_[bins_pixel_y_ht, bins_pixel_y_ht2]
    bins_pixel_y = np.r_[-bins_pixel_y_ht3[::-1], bins_pixel_y_ht3]

    return bins_pixel_x, bins_pixel_y


def pixel_bins_mid():
    ps = 4.434

    bins_pixel_x_ht = np.arange(-308.173, -3, ps)
    bins_pixel_x_ht2 = -np.arange(-308.173, -3, ps)[::-1]
    bins_pixel_x = np.r_[bins_pixel_x_ht, bins_pixel_x_ht2]

    bins_pixel_y_ht = -np.arange(-303.739, -3, ps)[::-1]
    bins_pixel_y_ht2 = np.arange(316.661, 615, ps)
    bins_pixel_y_ht3 = np.r_[bins_pixel_y_ht, bins_pixel_y_ht2]
    bins_pixel_y = np.r_[-bins_pixel_y_ht3[::-1], bins_pixel_y_ht3]

    return bins_pixel_x, bins_pixel_y

def pixel_clean_up(df, df_ht, borderw=0):
    df_clean_pixel = df.copy()

    border = borderw
    x_min = np.min(np.min(df_ht[['xh_i', 'xh_f']])) - border
    x_max = np.max(np.max(df_ht[['xh_i', 'xh_f']])) + border
    y_min = np.min(np.min(df_ht[['yh_i', 'yh_f']])) - border
    y_max = np.max(np.max(df_ht[['yh_i', 'yh_f']])) + border

    df_clean_pixel = df_clean_pixel[df_clean_pixel['x'] > x_min]
    df_clean_pixel = df_clean_pixel[df_clean_pixel['x'] < x_max]
    df_clean_pixel = df_clean_pixel[df_clean_pixel['y'] > y_min]
    df_clean_pixel = df_clean_pixel[df_clean_pixel['y'] < y_max]

    return df_clean_pixel

def pixel_bins_mid_test():
    ps = 4.434

    bins_pixel_x_ht = np.arange(-308.173, -3, ps)
    bins_pixel_x_ht2 = -np.arange(-308.173, -3, ps)[::-1]
    bins_pixel_x = np.r_[bins_pixel_x_ht, bins_pixel_x_ht2]

    bins_pixel_y_ht = -np.arange(-308.173, -3, ps)[::-1]
    bins_pixel_y_ht2 = np.arange(316.661, 612, ps)
    bins_pixel_y_ht3 = np.r_[bins_pixel_y_ht, bins_pixel_y_ht2]
    bins_pixel_y = np.r_[-bins_pixel_y_ht3[::-1], bins_pixel_y_ht3]

    return bins_pixel_x, bins_pixel_y

def pixel_bins_edge_test():
    ps = 4.434

    bins_pixel_x_ht = np.arange(-308.173 - ps/2, -3, ps)
    bins_pixel_x_ht2 = -np.arange(-308.173 - ps/2, -3, ps)[::-1]
    bins_pixel_x = np.r_[bins_pixel_x_ht, bins_pixel_x_ht2]

    bins_pixel_y_ht = -np.arange(-310.39, -3, ps)[::-1]
    bins_pixel_y_ht2 = np.arange(312.227 + ps/2, 615, ps)
    bins_pixel_y_ht3 = np.r_[bins_pixel_y_ht, bins_pixel_y_ht2]
    bins_pixel_y = np.r_[-bins_pixel_y_ht3[::-1], bins_pixel_y_ht3]

    return bins_pixel_x, bins_pixel_y

def pixel_bins_mid_test_mod1():
    ps = 4.434

    bins_pixel_x_ht = np.arange(-308.163, -1, ps)
    bins_pixel_x_ht2 = -np.arange(-308.163, -1, ps)[::-1]
    bins_pixel_x = np.r_[bins_pixel_x_ht, bins_pixel_x_ht2]

    bins_pixel_y_ht = -np.arange(-303.729, -3, ps)[::-1]
    bins_pixel_y_ht2 = np.arange(317.031, 615, ps)
    bins_pixel_y_ht3 = np.r_[bins_pixel_y_ht, bins_pixel_y_ht2]
    bins_pixel_y = np.r_[-bins_pixel_y_ht3[::-1], bins_pixel_y_ht3]

    return bins_pixel_x, bins_pixel_y

def pixel_bins_mid_test_mod2():
    ps = 3.8

    bins_pixel_x_ht = np.arange(-305.29, -4, ps)
    bins_pixel_x_ht2 = -np.arange(-305.29, -4, ps)[::-1]
    bins_pixel_x = np.r_[bins_pixel_x_ht, bins_pixel_x_ht2]

    bins_pixel_y_ht = -np.arange(-305.29, -5, ps)[::-1]
    bins_pixel_y_ht2 = np.arange(315.47, 616, ps)
    bins_pixel_y_ht3 = np.r_[bins_pixel_y_ht, bins_pixel_y_ht2]
    bins_pixel_y = np.r_[-bins_pixel_y_ht3[::-1], bins_pixel_y_ht3]

    return bins_pixel_x, bins_pixel_y

def pixel_bins_mid_test_mod3():
    ps = 4.434

    bins_pixel_x_ht = np.arange(-308.163, -1, ps)
    bins_pixel_x_ht2 = -np.arange(-308.163, -1, ps)[::-1]
    bins_pixel_x = np.r_[bins_pixel_x_ht, 0, bins_pixel_x_ht2]

    bins_pixel_y_ht = -np.arange(-303.729, -1, ps)[::-1]
    bins_pixel_y_ht2 = np.arange(317.031, 615, ps)
    bins_pixel_y_ht3 = np.r_[bins_pixel_y_ht, bins_pixel_y_ht2]
    bins_pixel_y = np.r_[-bins_pixel_y_ht3[::-1], 0, bins_pixel_y_ht3]

    return bins_pixel_x, bins_pixel_y

def pixel_bins_edge_test_mod1():
    ps = 4.434

    bins_pixel_x_ht = np.arange(-308.163 - ps/2, -3, ps)
    bins_pixel_x_ht2 = -np.arange(-308.163 - ps/2, -3, ps)[::-1]
    bins_pixel_x = np.r_[bins_pixel_x_ht, bins_pixel_x_ht2]

    bins_pixel_y_ht = -np.arange(-305.946, -3, ps)[::-1]
    bins_pixel_y_ht2 = np.arange(314.814, 615, ps)
    bins_pixel_y_ht3 = np.r_[bins_pixel_y_ht, bins_pixel_y_ht2]
    bins_pixel_y = np.r_[-bins_pixel_y_ht3[::-1], bins_pixel_y_ht3]

    return bins_pixel_x, bins_pixel_y

def pixel_bins_edge_test_mod2():
    ps = 3.8

    bins_pixel_x_ht = np.arange(-305.29 - ps/2, -3, ps)
    bins_pixel_x_ht2 = -np.arange(-305.29 -ps/2, -3, ps)[::-1]
    bins_pixel_x = np.r_[bins_pixel_x_ht, bins_pixel_x_ht2]

    bins_pixel_y_ht = -np.arange(-307.19, -3, ps)[::-1]
    bins_pixel_y_ht2 = np.arange(313.57, 618, ps)
    bins_pixel_y_ht3 = np.r_[bins_pixel_y_ht, bins_pixel_y_ht2]
    bins_pixel_y = np.r_[-bins_pixel_y_ht3[::-1], bins_pixel_y_ht3]

    return bins_pixel_x, bins_pixel_y

def pixel_bins_edge_test_mod3():
    ps = 4.434

    bins_pixel_x_ht = np.arange(-308.163 - ps/2, -1, ps)
    bins_pixel_x_ht2 = -np.arange(-308.163 -ps/2, -1, ps)[::-1]
    bins_pixel_x = np.r_[bins_pixel_x_ht, 0, bins_pixel_x_ht2]

    bins_pixel_y_ht = -np.arange(-614.109 - (4.434/2), 0, ps)[::-1]
    bins_pixel_y = np.r_[-bins_pixel_y_ht[::-1], 0, bins_pixel_y_ht]

    return bins_pixel_x, bins_pixel_y

def pixel_bins_z_mid():

    ps = 4.434

    bins_pixels_z_ht = np.arange(0, 605, ps)

    return bins_pixels_z_ht

def pixel_bins_mid_z_mod2():

    ps = 3.8

    bins_pixels_z_ht = np.arange(0, 605, ps)

    return bins_pixels_z_ht


def get_pixels_subset(filename, subset_cat, k=10):

    if filename.__contains__('datalog') or filename.__contains__('larndsim'):
        bins_pixel_x_edge, bins_pixel_y_edge = pixel_bins_mid_test()
        mod = 'mod0'

    if filename.__contains__('event'):
        bins_pixel_x_edge, bins_pixel_y_edge = pixel_bins_mid_test_mod1()
        mod = 'mod1'

    if filename.__contains__('self_trigger'):
        bins_pixel_x_edge, bins_pixel_y_edge = pixel_bins_mid_test_mod2()
        mod = 'mod2'

    if filename.__contains__('tpc12-reco'):
        bins_pixel_x_edge, bins_pixel_y_edge = pixel_bins_mid_test_mod3()
        mod = 'mod3'

    if subset_cat == 'cat1':

        # _|     |_ The corners of the entire pixel plane

        bins_x = np.r_[bins_pixel_x_edge[:k], bins_pixel_x_edge[-k:]]
        bins_y = np.r_[bins_pixel_y_edge[:k], bins_pixel_y_edge[-k:]]
        return bins_x, bins_y

    if subset_cat == 'cat2':

        # |﹉﹉﹉|, |﹍﹍﹍| The pixel edges on the top and bottom

        bins_x = np.r_[bins_pixel_x_edge[k:-k]]
        bins_y = np.r_[bins_pixel_y_edge[:k], bins_pixel_y_edge[-k:]]

        return bins_x, bins_y

    if subset_cat == 'cat3':

        # The pixel edges on the sides
        # |     |
        # |     |
        # |     |

        bins_x = np.r_[bins_pixel_x_edge[:k], bins_pixel_x_edge[-k:]]
        bins_y = np.r_[bins_pixel_y_edge[k:-k]]

        return bins_x, bins_y

    if subset_cat == 'cat4':

        # |﹉﹉﹉| Tile center, which is k pixels from the edge of the tile (x4 in y, x2 in x)
        # |﹍﹍﹍|

        half_way_x = int(len(bins_pixel_x_edge)/2)
        bins_x = np.r_[bins_pixel_x_edge[k:half_way_x-k], bins_pixel_x_edge[half_way_x+k:-k]]

        quarter_way_y = int(len(np.unique(np.abs(bins_pixel_y_edge)))/2)
        half_way_y = int(len(bins_pixel_y_edge)/2)

        bins_y = np.r_[bins_pixel_y_edge[k:quarter_way_y-k-1], bins_pixel_y_edge[quarter_way_y+k-1:half_way_y-k],
        bins_pixel_y_edge[half_way_y+k:-quarter_way_y-k+1], bins_pixel_y_edge[-quarter_way_y+k+1:-k]]

        bins_y = np.r_[bins_pixel_y_edge[k:-k]]

        return bins_x, bins_y

    if subset_cat == 'cat5':

        bins_x = np.r_[bins_pixel_x_edge[k:-k]]

        # |﹍﹍﹍|
        # |﹉﹉﹉| This is the boundary we are trying to describe between each row of pixel tiles (x3)

        # To find the position of the top gap (half-way across the positive pixel positions in y)
        half_way = int(len(np.unique(np.abs(bins_pixel_y_edge)))/2)

        # Find the k surrounding pixels
        bot_top = np.unique(np.abs(bins_pixel_y_edge))[half_way+1-k:half_way+1]
        bot_bot = np.unique(np.abs(bins_pixel_y_edge))[half_way+1:half_way+1+k]

        bins_y = np.r_[-np.unique(np.abs(bins_pixel_y_edge))[:k], np.unique(np.abs(bins_pixel_y_edge))[:k],
        bot_top, bot_bot, -bot_top, -bot_bot]

        return bins_x, bins_y

    if subset_cat == 'cat6':

        # ﹍﹍| |﹍﹍
        # ﹉﹉| |﹉﹉ The vertical boundary between the pixel tiles (x1)

        bins_x = np.r_[-np.unique(np.abs(bins_pixel_x_edge))[:k], np.unique(np.abs(bins_pixel_x_edge))[:k]]
        bins_y = bins_pixel_y_edge

        return bins_x, bins_y
