import numpy as np
from cmath import pi

#get_resolution(level: number): number {
#    return this._computed_initial_resolution() / 2**level
#  }
sf=(1.00301/0.5015)#*(vstate.maxds/32.0063)#*(vstate.maxds/32.0063)
init_res=40211.5*sf*(2/(100*pi))
min_zoom=0
max_zoom=10
resolutions = [init_res/2**lev for lev in range(min_zoom, max_zoom+1)]

def get_level_by_extent(extent):
    
    x_rs = (extent[2] - extent[0]) / 1700
    y_rs = (extent[3] - extent[1]) / 1000
    resolution = np.maximum(x_rs, y_rs)

    i = 0
    for r in resolutions:
      if resolution > r:
        if i == 0:
          return 0
        if i > 0:
          return i - 1
      i += 1

    #otherwise return the highest available resolution
    return (i-1)
  


