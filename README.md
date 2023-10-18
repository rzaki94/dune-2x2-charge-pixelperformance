# dune-2x2-charge-pixelperformance

Designed to study the pixel performance in the 2x2 ArgonCube prototype at Fermilab & run in-situ diagnostics

Links to NERSC file storage for individual modules:

mod0_dir = '/global/cfs/projectdirs/dune/www/data/Module0/TPC1+2/dataRuns/tracksData/'

mod1_dir = '/global/cfs/projectdirs/dune/www/data/Module1/reco/charge_only/'

mod2_dir = '/global/cfs/projectdirs/dune/www/data/Module2/charge_reco/v1/TPC12_run2/'

mod3_dir = '/global/cfs/projectdirs/dune/www/data/Module3/run3/reco/tpc12/'

mod_sim  = '/global/cfs/projectdirs/dune/www/data/Module0/simulation/larndsim/20211223/reco/'

# Explanation of scripts

# 1. The conversion script: h5_to_pkl_modules.py

The data output of each individual module is stored in h5 file format and contains the pixel hit information (**keep in mind to use the reco.h5 files**
  Hits information
  
  - Deposited charge
  - X, Y & electronical position (io_group, io_channel, chip_id, channel_id)
  - Hit timing
    
  Event information:
  - Event start time
  - Number of hits in events

  The event time is used to understand the drift time for individual hits in order to reconstruct the drift distance and thus the 3rd coordinate of the track's position

Output: 2 files for each drift volume containing: ['px', 'py', 'ts', 'q'] (px = x position, py = y position, ts = drift time since event start, q = deposited charge)

# 2. The Hough line algorithm script: get_Hough.py

Now that the x, y & time for each hit is stored and the files have been trimmed, tracks can be found for each individual event with a hough line transformation (HL) algorithm

The leading function: GetLinesEvents
  - subfunctions found in "houghutils.py"
    - GetLinesEvent: Gets the lines for each event
    - HoughOnArray: Actually performs the hough line transform algorithm
    - d_pnt2line: Calculates the distance between the hough line and the hits (filters out > radius)
    - trackLengthhoughcenter: calculates the hough line center based on the associated hits and track length
    - set_newhlcenter: sets the new center 

  - Assigns the radius around the main ionization line that hits should be saved in
    - Hits should be within drift volume (z > 0, z < 315)
      
  - Loops over all events and finds track that fit the following **tunable** requirements:
    - A minimum of 100 points in the hit cloud to be identified as a line, maximum of 5 lines found in an event
    - Track passes through 2 of the 6 detector faces
    - The ratio between energy contained within the respective radius and a cylinder of 25 mm should be > 0.9 to filter for MIP-like particles
    - dQ/dX should be smaller than 3
    
  - The output still has the event by event structure as this is necessary for future manipulation
    - List of dataframes of hough lines ['aX', 'aY', 'aZ', 'bX', 'bY', 'bZ', 'trackL'] (a* = midpoints, b* = angles, track length) &
    - associated event ['start_X', 'start_Y', 'start_Z', 'EnergyDeposit', 'd_hl_0'] (start_* = position, energy deposit, distance between hit and hough line "0")
   
# 3. The pixel performance script: getRandomSeg.py

The framework is set up to find the pixel efficiencies. There are several methods of plotting the performance, which justify the information that we store in the output file. 

![image](https://github.com/rzaki94/dune-2x2-charge-pixelperformance/assets/60906881/e5e53262-90dd-4472-a23d-489280653020)


The leading function: findpixelefficiency

TBC


