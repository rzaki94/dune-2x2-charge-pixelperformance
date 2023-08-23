# dune-2x2-charge-pixelperformance

Designed to study the pixel performance in the 2x2 ArgonCube prototype at Fermilab & run in-situ diagnostics

# Explanation of scripts

# 1. The conversion script: event_to_pkl_modules.py

The data output of each individual module is stored in h5 file format and contains the pixel hit information
  Hits information
  
  - Deposited charge
  - X, Y & electronical position (io_group, io_channel, chip_id, channel_id)
  - Hit timing
    
  Event information:
  - Event start time
  - Number of hits in events

  The event time is used to understand the drift time for individual hits in order to reconstruct the drift distance and thus the 3rd coordinate of the track's position
