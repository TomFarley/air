function raw2temp_aug,sh,d,integtime, aug=aug

;calibrate the camera using the lab calibration and the invessel heated
;tile (emissivity)

;data for number of photons at a given temperature in correct wavelength range
OPENR, lun, './calib/planckBB.dat', /GET_LUN

bbtable = fltarr(2,601)
READF, lun, bbtable

;initialise variables
Tpro=d*0.0
nrows=n_elements(bbtable(0,*))
temp=float(bbtable(0,*))
photon=float(bbtable(1,*))
c1=0
c2=0

;Read in the calibration factors for the camera (from lab calib)
calib_vals=read_ascii('calib/calib_coeffs.dat', data_start=6)

;find the correct vals based on the shot number (sh)
locate_sh=where(calib_vals.field1[0,*] lt sh)
calib_values=reform(calib_vals.field1[*, n_elements(locate_sh)-1])

;stop

c1=(calib_values[1])/(integtime)+calib_values[2]
c2=(calib_values[3])/(integtime)+calib_values[4]
trans_correction=calib_values[5] ;window transmission correction (>1 as window
								;attenuates)

;convert counts to photons. Include the attenuation caused by the window
planck=c2*(d*trans_correction)
bgtemp=23 ;background temperature

;use the data from bbtable.dat to convert from recorded photons to temp.
;need to add on the photons which are the background (NUC makes zero ADC counts
;the background temp
bgphot=interpol(photon,temp,bgtemp)

;then determine the temperature using the photon counts plus the bckg.
Tpro=interpol(temp,photon,planck+bgphot)

free_lun, lun

return, Tpro

;Saved calibration factors - should appear in ../calib_coeffs.dat
;c1=(8.09e13)/(integtime)-1.09e17 ;Greg's last calib
;c2=(5.1e9)/(integtime)-1.3e13
;c1=((8.45e13)/integtime)-5.12e17 ;AJT calib Oct '10
;c2=((4.84e9)/integtime)+6.00e13

end
