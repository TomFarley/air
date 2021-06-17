import numpy as np
import matplotlib.pyplot as plt
import pyEquilibrium
from pyEquilibrium import equilibrium
import pyuda
client = pyuda.Client()


def wettedFraction(trace, shot, start, end):
    """ Return the wetted fraction of the specified divertor tile at the
    specified time

    trace:  One of the four 'AI*_ETOTSUM_*SP' traces from UDA
    shot:   MAST shot number (must be an integer)
    start:  Beginning of the time interval (in seconds)
    end:    End of the time interval
    """
    # Read the Qprofile trace corresponding to the ETOTSUM trace
    letters = np.empty(15)
    letters = trace
    Qprofile = client.get('AI' + letters[2] + '_QPROFILE_' + letters[12] +
                          'SP', shot)
    QprofileData = np.transpose(Qprofile.data)
    #timeSamples = np.linspace(start, end, 6)
    #f_wet = []

    #for i in range(len(timeSamples)-1):
    #start = timeSamples[i]
    #end = timeSamples[i+1]
    argEnd = np.argmax(Qprofile.dims[0].data > end)
    argStart = np.argmax(Qprofile.dims[0].data > start)
    #print(Qprofile.dims)
    #print(Qprofile.dims[0].data[argStart])
    #print(Qprofile.dims[0].data[argEnd])

    # Find radius of peak heat flux for heat flux profile averaged over time window
    meanSlice = np.mean(QprofileData[:, argStart:argEnd], axis=1)
    radius = np.abs(Qprofile.dims[1].data[np.argmax(meanSlice)])
    time = (start+end)/2
    # Coordinates for upper and lower cameras
    if letters[2] == 'R':  # AIR bottom, AIT top
        zCoord = -1.825
    else:
        zCoord = 1.825

    # Read the magnetic data
    # eqData = pyEquilibrium.equilibrium('MAST', shot, time)
    eqData = equilibrium.equilibrium('MAST', shot, time)
    B_Z = eqData.BZ
    B_T = eqData.Bt
    B_R = eqData.BR
    bz = B_Z(radius, zCoord)[0][0]
    bt = B_T(radius, zCoord)[0][0]
    br = B_R(radius, zCoord)[0][0]

    # Why don’t you take the arcsin? (I realise given the small angle approximation it doesn’t make much of a difference in this case!)
    fieldAngle = (-bz)/(np.sqrt(bt**2 + br**2 + bz**2))
    print('field_angle:', fieldAngle)
    print('field_angle deg:', np.rad2deg(fieldAngle))
    print('arcsin angle:', np.arcsin(fieldAngle))

    f_wet = (np.sin(fieldAngle)
             / (np.sin(4/360*2*np.pi + fieldAngle)))

#    plt.contourf(Qprofile.dims[0].data, Qprofile.dims[1].data,
#                 QprofileData, levels=np.linspace(np.min(QprofileData),
#                                                  np.max(QprofileData),
#                                                  100))
#    plt.title('Q profile [' + Qprofile.units + '] shot ' + str(shot))
#    plt.colorbar()
#    plt.axvline(start, color='red')
#    plt.axvline(start + 0.05, color='red')
#    plt.ylabel('radius [metres]')
#    plt.xlabel('time [seconds]')
#    plt.show()
#
#    plt.contourf(Qprofile.dims[0].data[argStart:argEnd],
#                 Qprofile.dims[1].data, QprofileData[:, argStart:argEnd],
#                 levels=np.linspace(np.min(QprofileData),
#                                    np.max(QprofileData), 100))
#    plt.colorbar()
#    plt.ylabel('radius [metres]')
#    plt.xlabel('time [seconds]')
#    plt.show()
#
#    plt.plot(Qprofile.dims[1].data, meanSlice)
#    plt.axvline(radius, color='red')
#    plt.xlabel('radius [metres]')
#    plt.ylabel('heat flux [MW m^-2]')
#    plt.title('heat flux averaged over time interval, R = ' + str(radius))
#    plt.show()
#
#    plt.contourf(eqData.R, eqData.Z, B_Z,
#                 levels=np.linspace(np.min(B_Z), np.max(B_Z), 100))
#    plt.title('BZ' + ' shot ' + str(shot) + ' @ ' + str(time*1000) + 'ms')
#    plt.colorbar()
#    plt.xlabel('R [m]')
#    plt.ylabel('Z [m]')
#    plt.plot(radius, -1.825, 'x', color='black')
#    plt.show()
#
#    plt.contourf(eqData.R, eqData.Z, B_T,
#                 levels=np.linspace(np.min(B_T), np.max(B_T), 100))
#    plt.title('BT' + ' shot ' + str(shot) + ' @ ' + str(time*1000) + 'ms')
#    plt.colorbar()
#    plt.xlabel('R [m]')
#    plt.ylabel('Z [m]')
#    plt.plot(radius, -1.825, 'x', color='black')
#    plt.show()
#
#    plt.contourf(eqData.R, eqData.Z, B_R,
#                 levels=np.linspace(np.min(B_R), np.max(B_R), 100))
#    # plt.plot(eqData.R[rCoord], eqData.Z[zCoord], 'x', color='red',
#               markersize=10, label='lower outer')
#    plt.title('BR' + ' shot ' + str(shot) + ' @ ' + str(time*1000) + 'ms')
#    plt.colorbar()
#    plt.xlabel('R [m]')
#    plt.ylabel('Z [m]')
#    plt.plot(radius, -1.825, 'bx', color='black')
#    plt.show()

    return f_wet


print('fwt:', wettedFraction('AIR_ETOTSUM_OSP', 27909, 0.16, 0.21))
