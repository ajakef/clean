st = obspy.read('data/noise.mseed')
st.filter('bandpass', freqmin=0.15, freqmax = 1)
inv = obspy.read_inventory('data/XP_PARK_inventory.xml') # includes coordinates
clean.add_inv_coords(st, inv) # store the coordinates in the stream

## define slowness grid to search
s_list = np.arange(-4, 4, 0.25)
result = clean.clean(st, phi = 0.1, win_length_sec = 180, 
                     freq_bin_width = 1, freq_min = 0, freq_max = 2, 
                     sxList = s_list, syList = s_list, prewhiten = False)
plt.figure(3)
plt.subplot(2,2,1)
clean.plot_freq_slow_spec(result, 'xy', 'original')
plt.subplot(2,2,2)
clean.plot_freq_slow_spec(result, 'xy')
plt.subplot(2,2,3)
clean.polar_freq_slow_spec(result, 'fh')
plt.subplot(2,2,4)
clean.polar_freq_slow_spec(result, 'fa')
plt.tight_layout()


