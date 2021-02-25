# "Clean" Beamforming
Clean is a method to iteratively deconvolve an array's response from the array's cross-spectrum, resulting in an output frequency-slowness spectrum that consists of a small number of spikes in slowness space. The beamforming code in this package was written following [den Ouden et al. (2020)](https://academic.oup.com/gji/article/221/1/305/5698307?casa_token=njSpOoXp9ekAAAAA:Cl9dOheqodGD02UsipR9peku690_jNEwcikjjkpHgxF3mjW-51NUwoFj3xIuQk6UPtfZN1tUYWfo) who used it for infrasound analysis; however, the Clean beamforming method has been in use for decades.

### Installation
Dependencies include pandas and obspy (essential) and mtspec (optional; supports multitaper spectra). You can make a conda environment with these dependencies using this command:
conda create -y -n clean_beamform python=3.8 pandas obspy mtspec

Then, download the repository either by clicking the creen "Code" button near the top right followed by "Download Zip", or by "git clone https://github.com/ajakef/clean.git".

Finally, run the demos or tests functions. The demos show how it can be used. The tests are supposed to show that it does actually work. I suggest using a tool that supports cells like Spyder for running the code.

### Incomplete power recovery in coherent wavefields
This is a major question: in real data (but not synthetic data) with high signal-to-noise ratio and simple wavefields, why is the clean power not approximately equal to the total power? For example, see the primary earthquake infrasound in demo_real_data.py: peak magnitude-squared semblance is 0.8, but the clean spectrum only has about 60% of the total power. 

### Negative diagonals in cross-spectrum:
This is another big open question: what to do if, after a clean iteration, one of the diagonal values in the cross spectrum is negative? The diagonal values represent power at that frequency at a given sensor; negative values are unphysical and can result in meaningless negative F-statistics. So dealing with this is a major problem. 

Currently, we just zero out a frequency as soon as one of its diagonal values goes negative. The logic here is that power shouldn't be negative if that frequency is dominated by plane waves hitting all stations, so if it does go negative after a wave is removed it means that power at that frequency is dominated by some non-wave noise and is no longer interesting.

This has the problem of being fragile to single-sensor problems. For example, if a sensor is muffled over certain frequencies, it will record lower powers than all the other sensors and will go negative while there is true wave energy remaining in the cross-spectrum. Conversely, if a sensor records higher noise than the others for some reason, it can result in wave power at certain frequencies being over-estimated, causing power to go negative for the normal sensors. Either would cause errors in the resulting frequency-slowness spectrum.

Ideally, we would have a more robust means of estimating wave power that handles outliers more gracefully.