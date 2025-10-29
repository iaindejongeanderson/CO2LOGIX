import numpy as np

def make_distance_matrix(x, y, R, gx, gy):
    
    # Mask of cells within R in x and y
    mask_x = (gx[0, :] >= x - R) & (gx[0, :] <= x + R)
    mask_y = (gy[:, 0] >= y - R) & (gy[:, 0] <= y + R)

    # Grid subset
    gx_local = gx[np.ix_(mask_y, mask_x)]
    gy_local = gy[np.ix_(mask_y, mask_x)]

    # Distance from (x, y) to local grid
    dist_local = np.sqrt((gx_local - x)**2 + (gy_local - y)**2)
    dist_local[dist_local==0]=0.1
    
    return dist_local, mask_y, mask_x

def estimate_frac_pres(sv,pp):
    # Based on the work of Zhang & Yin (2017) who found that on average
    # Fracture pressure was 0.75 x the difference between pp and sv

    return pp + (0.75 * (sv-pp))