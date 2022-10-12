import numpy as np
from numpy.random import normal as rnorm
import matplotlib.pyplot as plt
from numpy import pi, sqrt, sin, cos, exp
import obspy, scipy, importlib, sys
sys.path.append('/home/jake/Work/Aftershocks/lib/')
try:
    importlib.reload(cleanbf)
    importlib.reload(ProcessEvent)
    print('reloaded')
except:
    import cleanbf, ProcessEvent




#%%
## aftershock
t1 = obspy.UTCDateTime('2020-04-14T03:27:06.6')
eq_stream_all = ProcessEvent.ReadEventMS(t1, start = -81, end = 81, plotOnly=False, plot_t1 = np.NaN, plot_t2 = np.NaN,
                                         offsetFile = '/home/jake/Work/Aftershocks/ArrayProcessing/2020-04-14_offsets.txt',
                                         omit = ['084'], makeFrames = False, ds = 0.01, winLength = 1, winFrac = 0.2, 
                                         mseed_dir = '/home/jake/Work/Aftershocks/2020-04-24_PARK/mseed', 
                                         locFile = '/home/jake/Work/Aftershocks/project_coords.csv',
                                         fl = np.nan, fh = np.nan)
eq_stream_all.write('/home/jake/Work/Aftershocks/lib/clean/data/aftershock.mseed')



#%%
## long noise period before aftershock
t1 = obspy.UTCDateTime('2020-04-14T03:27:06.6')
eq_stream_all = ProcessEvent.ReadEventMS(t1, start = -910, end = -10, plotOnly=False, plot_t1 = np.NaN, plot_t2 = np.NaN,
                                         offsetFile = '/home/jake/Work/Aftershocks/ArrayProcessing/2020-04-14_offsets.txt',
                                         omit = ['084', '055'], makeFrames = False, ds = 0.01, winLength = 1, winFrac = 0.2, 
                                         mseed_dir = '/home/jake/Work/Aftershocks/2020-04-24_PARK/mseed', 
                                         locFile = '/home/jake/Work/Aftershocks/project_coords.csv',
                                         fl = np.nan, fh = np.nan)
eq_stream_all.write('/home/jake/Work/Aftershocks/lib/clean/data/noise.mseed')


## Old, not used

#%% Demonstrate that Clean definitely can work on real high-SNR PARK data. Omit noisy 084
eq_time = obspy.UTCDateTime('2020-05-15T11:03:27')
path = '/home/jake/Work/Aftershocks/2020-05-23_PARK/'
s_list = np.arange(-4, 4, 0.05) # this is forward slowness, not back-slowness
eq_stream=ProcessEvent.ReadEventMS(eq_time, start = 210, end = 270, omit = ['084'], fl = .1, fh = 1, # fl 0.2
                                   mseed_dir = path + 'mseed/', locFile = path + 'PARK_coords_2020-05-23.csv')

eq_stream.write('/home/jake/Work/Aftershocks/lib/clean/data/tonopah_primary.mseed')

