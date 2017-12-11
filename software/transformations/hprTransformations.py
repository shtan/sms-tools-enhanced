# functions that implement transformations using the hprModel

import numpy as np
from scipy.interpolate import interp1d

def hprTimeScale(hfreq, hmag, mXres, timeScaling):
    """
    Time scaling of the harmonic plus residual representation
    hfreq, hmag: harmonic frequencies and magnitudes, stocEnv: residual envelope
    timeScaling: scaling factors, in time-value pairs
    returns yhfreq, yhmag, ymXres: hps output representation
    """

    if (timeScaling.size % 2 != 0):                        # raise exception if array not even length
        raise ValueError("Time scaling array does not have an even size")
        
    L = hfreq[:,0].size                                    # number of input frames
    maxInTime = max(timeScaling[::2])                      # maximum value used as input times
    maxOutTime = max(timeScaling[1::2])                    # maximum value used in output times
    outL = int(L*maxOutTime/maxInTime)                     # number of output frames
    inFrames = (L-1)*timeScaling[::2]/maxInTime            # input time values in frames
    outFrames = outL*timeScaling[1::2]/maxOutTime          # output time values in frames
    timeScalingEnv = interp1d(outFrames, inFrames, fill_value=0)    # interpolation function
    indexes = timeScalingEnv(np.arange(outL))              # generate frame indexes for the output
    yhfreq = hfreq[int(round(indexes[0])),:]                    # first output frame
    yhmag = hmag[int(round(indexes[0])),:]                      # first output frame
    ymXres = mXres[int(round(indexes[0])),:]                # first output frame
    for l in indexes[1:]:                                  # iterate over all output frame indexes
        yhfreq = np.vstack((yhfreq, hfreq[int(round(l)),:]))      # get the closest input frame
        yhmag = np.vstack((yhmag, hmag[int(round(l)),:]))         # get the closest input frame
        ymXres = np.vstack((ymXres, mXres[int(round(l)),:])) # get the closest input frame
    return yhfreq, yhmag, ymXres
    
def hprResidualTimbreScale(mXres, changeResEnv, timbreScalingRes, maxFreqFrac):
    """
    Scales formants of residual representation
    changeResEnv: boolean; if 0 leave it as it is, if 1 change
    timbreScalingRes: scaling factor
    maxFreqFrac: above this frequency, don't apply scaling
    """
    L = mXres.shape[0]                                      # number of frames
    for l in range(L):
        mXr = mXres[l,:]
        if (changeResEnv == 1):
            mXrshift = np.zeros(mXr.size)
            for i in range(0, mXr.size):
                #if (round(float(i)/0.88)<mX.size):
                if (round(float(i)/timbreScalingRes)<mXr.size and (i/mXr.size)<maxFreqFrac):
                    mXrshift[i] = mXr[int(round(float(i)/timbreScalingRes))]
                else:
                    mXrshift[i] = mXr[i]
        if l == 0:                                          # if first frame
            ymXres = np.array([mXrshift])
        else:                                               # rest of frames
            ymXres = np.vstack((ymXres, np.array([mXrshift])))
    return ymXres