#POLAR PLOTTING:
def calc_mean_and_vector_strength(position_dt):
    vec_strength = 1-circvar(position_dt)
    mean_drxn=circmean(position_dt)  
    return vec_strength, mean_drxn

def circvar(alpha,axis=None):
    alpha= np.deg2rad(alpha)
    if np.ma.isMaskedArray(alpha) and alpha.mask.shape!=():
        N = np.sum(~alpha.mask,axis)
    else:
        if axis is None:
            N = alpha.size
        else:
            N = alpha.shape[axis]
    R = np.sqrt(np.sum(np.sin(alpha),axis)**2 + np.sum(np.cos(alpha),axis)**2)/N
    V = 1-R
    return V
# Circular mean heading
def circmean(alpha,axis=None ):
    # for val in range(len(alpha)):
    #     if alpha[val] < 0:
    #         alpha[val] +=360
    alpha= np.deg2rad(alpha)
    mean_angle = np.arctan2(np.nanmean(np.sin(alpha),axis),np.nanmean(np.cos(alpha),axis))
    if mean_angle<0:
        mean_angle=mean_angle+2*np.pi
    return mean_angle
def deg360to180(angle):
    if angle <0:
        angle %=360
    elif angle > 180:
        angle = 360 - angle
    
    return angle