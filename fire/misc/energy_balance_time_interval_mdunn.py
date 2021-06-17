import numpy as np
import matplotlib.pyplot as plt
import pyuda
from scipy import integrate
from wetted_area_mdunn import wettedFraction
client = pyuda.Client()

timeLength = 0.05		# Length of time interval
wettedArea = True

# Which data traces do we want to read?
tracenames = ['EFM_PLASMA_ENERGY', 'ESM_X', 'ABM_PRAD_POL', 'ESM_P_PHI',
              'ESM_P_AUX', 'AIR_ETOTSUM_OSP', 'AIR_ETOTSUM_ISP',
              'AIT_ETOTSUM_OSP', 'AIT_ETOTSUM_ISP']
shortnames = ['thermal', 'magnetic', 'radiated', 'ohmic',
              'NBI', 'lower outer', 'lower inner',
              'upper outer', 'upper inner']
# Identify as 0 for energy sources, 1 for stored energy and 2 for energy sinks
identifier = [1, 1, 2, 0,
              0, 2, 2,
              2, 2]

# Load the data
shots = [26394, 27775, 28322]
# shotlist = np.loadtxt('mast_power_balance_shot_lists/trial_shot_list_15.txt', dtype=int)
shotlist = np.loadtxt('mast_power_balance_shot_lists/trial_shot_list_12.txt', dtype=int)
# timelist = np.loadtxt('mast_power_balance_shot_lists/time_windows_15_all_0.05.txt')
timelist = np.loadtxt('mast_power_balance_shot_lists/time_windows_12_all_0.05.txt')
# indices = np.load('near mean values.npy')[1]
indices = []
for thisShot in shots:
    indices.append(np.where(shotlist == thisShot)[0][0])
# indices = np.array(range(10), dtype=int) + 10
# print(indices)

shotlist = shotlist[indices]    # Limited shots only
timelist = timelist[indices]
# print(timelist)

for i in range(len(indices)):
    print(shotlist[i], indices[i], timelist[i])

power_balance = []
plotdata = []

# Loop over each shot
for i, shot in enumerate(shotlist):
    print('shot', i+1, '/', len(shotlist), 'has been processed')
    start = timelist[i]
    end = start + timeLength

    readdata = []		# Arrays for data before and after procesing
    plotdata.append([])

    # Read in the data from UDA
    for m, trace in enumerate(tracenames):
        readdata = client.get(trace, shot)
        timedata = np.where(np.logical_and(readdata.time.data > start,
                                           readdata.time.data < end))[0]
        interval = np.array([start] + readdata.time.data[timedata].tolist()
                            + [end])*1e3
        endData = np.interp([start, end], readdata.time.data, readdata.data)
        intervalData = np.array([endData[0]] + readdata.data[timedata].tolist()
                                + [endData[1]])
        # Account for the wetted area
        if m > 4 and wettedArea:
            intervalData = intervalData*wettedFraction(trace, shot, start,
                                                       end)
            outData = []
            for n in range(5):
                inData = intervalData[n*len(intervalData)//5:
                                      (n+1)*len(intervalData)//5]
                thisStart = start + n*timeLength/5
                thisEnd = start + (n+1)*timeLength/5
                outData.append(inData*wettedFraction(trace, shot,
                                                     thisStart, thisEnd))
            outData = np.asarray(outData)
            intervalData = [val for sublist in outData for val in sublist]
        # Convert each trace into kJ for plotting
        # Noting that 'interval' is in milliseconds not seconds
        if readdata.units == 'MW':
            plotdata[i].append(integrate.trapz(intervalData, interval))
        elif readdata.units == 'W':
            plotdata[i].append(integrate.trapz(intervalData/1e6, interval))
        elif readdata.units == 'J':
            plotdata[i].append((intervalData[-1] - intervalData[0])/1e3)
        elif readdata.units == 'kJ':
            plotdata[i].append(intervalData[-1] - intervalData[0])
        else:
            print('Units error for', trace, 'with units', readdata.units)
            print('Acceptable units are MW, W, J or kJ')
    '''
    #print(plotdata)
    for j, k in enumerate(identifier):
        if k != 0:
            plotdata[j] = -plotdata[j]
    #print(plotdata)
    power_balance.append(np.sum(plotdata))
    print('For shot', shot, 'energy discrepancy is', power_balance[i])
    '''

# Make a stacked bar chart with 2 sub-stacks per shot (sources & sinks)
xpos = np.arange(len(shotlist))
plotdata = np.array(plotdata)
p = []
base = np.zeros((len(shotlist), 3))

for j, k in enumerate(identifier):
    # If it is a stored energy trace, count as a source(sink) if it is negative
    # (positive) over the interval
    if k == 1:
        decider = []
        for m in range(len(shotlist)):
            if plotdata[m, j] > 0:
                decider.append(2)
                base[m, 1] = base[m, 2]
                base[m, 2] = base[m, 2] + plotdata[m, j]
            else:
                plotdata[m, j] = -plotdata[m, j]
                decider.append(0)
                base[m, 1] = base[m, 0]
                base[m, 0] = base[m, 0] + plotdata[m, j]
        decider = np.array(decider)
        p.append(plt.bar(xpos + (decider[:]-1)*0.2, np.abs(plotdata[:, j]),
                 width=0.4, bottom=base[:, 1]))
    # Plot the obvious sources and sinks
    else:
        p.append(plt.bar(xpos + (k-1)*0.2, plotdata[:, j], width=0.4,
                 bottom=base[:, k]))
        base[:, k] = base[:, k] + plotdata[:, j]

title = 'Energy sources and sinks in ' + str(timeLength) + 's time intervals'
plt.xticks(xpos, shotlist, rotation=45)
plt.xlabel('shot number')
plt.ylabel('Energy [kJ]')
plt.title(title)
# Use the bbox keyword to move the legend around with (x, y) coordinates
plt.legend(p, shortnames, bbox_to_anchor=(0.6, 0.5))
# plt.savefig('wet_time_outliers_15.png')
plt.show()
