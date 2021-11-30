#!/usr/bin/env python

"""


Created: 
"""

import logging, os, datetime
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.propagate = False

def ptw_to_dict(path_fn_ptw):
    from fire.interfaces import ryptw
    # os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data/functions')

    header = ryptw.readPTWHeader(path_fn_ptw)
    width = header.h_Cols
    height = header.h_Rows
    camera_SN = header.h_CameraSerialNumber
    NUCpresetUsed = header.h_NucTable
    FrameRate = 1/header.h_CEDIPAquisitionPeriod # Hz
    IntegrationTime = round(header.h_CEDIPIntegrationTime*1e6,0) # microseconds
    ExternalTrigger = None	# I couldn't find this signal in the header

    digitizer_ID = []
    data = []
    time_of_measurement = []
    frame_counter = []
    DetectorTemp = []
    SensorTemp_0 = []
    SensorTemp_3 = []
    AtmosphereTemp = []
    last = 0
    for i in range(header.h_firstframe,header.h_lastframe+1):
        frame = ryptw.getPTWFrames(header, [i])[0][0]
        if i == header.h_firstframe:
            new_shape = np.shape(frame.T)
        data.append(np.flip(frame.flatten().reshape(new_shape).T,axis=0))
        frame_header = ryptw.getPTWFrames(header, [i])[1][0]
        digitizer_ID.append(frame_header.h_framepointer%2)  # I couldn't find this so as a proxy, given the
        # digitisers are always alternated, I only use if the frame is even or odd
        # yyyy = frame_header.h_FileSaveYear
        # mm = frame_header.h_FileSaveMonth
        # dd = frame_header.h_FileSaveDay
        hh = frame_header.h_frameHour
        minutes = frame_header.h_frameMinute
        ss_sss = frame_header.h_frameSecond
        time_of_measurement.append((datetime.datetime(frame_header.h_FileSaveYear-30,frame_header.h_FileSaveMonth,
                                                      frame_header.h_FileSaveDay).timestamp() + hh*60*60 + minutes*60
                                    + ss_sss)*1e6)  # I leave if as a true timestamp, t=0 is 1970
        frame_counter.append(None)  # I couldn't find this in the header
        DetectorTemp.append(frame_header.h_detectorTemp)
        SensorTemp_0.append(frame_header.h_detectorTemp)	# this data is missing so I use the more similar again
        SensorTemp_3.append(frame_header.h_sensorTemp4)
        AtmosphereTemp.append(frame_header.h_AtmosphereTemp)	# additional data not present in the .ats format
    data = np.array(data)
    data_median = int(np.median(data))
    if np.abs(data-data_median).max()<2**8/2-1:
        data_minus_median = (data-data_median).astype(np.int8)
    elif np.abs(data-data_median).max()<2**16/2-1:
        data_minus_median = (data-data_median).astype(np.int16)
    elif np.abs(data-data_median).max()<2**32/2-1:
        data_minus_median = (data-data_median).astype(np.int32)
    digitizer_ID = np.array(digitizer_ID)
    time_of_measurement = np.array(time_of_measurement)
    frame_counter = np.array(frame_counter)
    DetectorTemp = np.array(DetectorTemp)
    SensorTemp_0 = np.array(SensorTemp_0)
    SensorTemp_3 = np.array(SensorTemp_3)
    AtmosphereTemp = np.array(AtmosphereTemp)
    out = dict([])
    # out['data'] = data
    out['data_median'] = data_median
    # I do this to save memory, also because the change in counts in a single recording is always small
    out['data'] = data_minus_median
    out['digitizer_ID'] = digitizer_ID
    out['time_of_measurement'] = time_of_measurement
    out['IntegrationTime'] = IntegrationTime
    out['FrameRate'] = FrameRate
    out['ExternalTrigger'] = ExternalTrigger
    out['NUCpresetUsed'] = NUCpresetUsed
    out['SensorTemp_0'] = SensorTemp_0
    out['SensorTemp_3'] = SensorTemp_3
    out['AtmosphereTemp'] = AtmosphereTemp
    out['DetectorTemp'] = DetectorTemp
    out['width'] = width
    out['height'] = height
    out['camera_SN'] = camera_SN
    out['frame_counter'] = frame_counter
    data_per_digitizer,uniques_digitizer_ID = separate_data_with_digitizer(out)
    out['data_time_avg_counts'] = np.array([(np.mean(data,axis=0)) for data in data_per_digitizer])
    out['data_time_avg_counts_std'] = np.array([(np.std(data,axis=0)) for data in data_per_digitizer])
    out['data_time_space_avg_counts'] = np.array([(np.mean(data,axis=(0,1,2))) for data in data_per_digitizer])
    out['data_time_space_avg_counts_std'] = np.array([(np.std(data,axis=(0,1,2))) for data in data_per_digitizer])
    out['uniques_digitizer_ID'] = uniques_digitizer_ID
    # return data,digitizer_ID,time_of_measurement,IntegrationTime,FrameRate,ExternalTrigger,SensorTemp_0,DetectorTemp,width,height,camera_SN,frame_counter
    return out

def separate_data_with_digitizer(full_saved_file_dict):
    try:
        data_median = full_saved_file_dict['data_median']
        data = full_saved_file_dict['data']
        data = data.astype(np.int)
        data += data_median
    except:
        data = full_saved_file_dict['data']
    digitizer_ID = full_saved_file_dict['digitizer_ID']
    uniques_digitizer_ID = np.sort(np.unique(digitizer_ID))
    data_per_digitizer = []
    for ID in uniques_digitizer_ID:
        data_per_digitizer.append(data[digitizer_ID==ID])
    return data_per_digitizer,uniques_digitizer_ID

def generic_separate_with_digitizer(data,digitizer_ID):
    uniques_digitizer_ID = np.sort(np.unique(digitizer_ID))
    data_per_digitizer = []
    for ID in uniques_digitizer_ID:
        data_per_digitizer.append(data[digitizer_ID==ID])
    return data_per_digitizer,uniques_digitizer_ID

if __name__ == '__main__':
    pass