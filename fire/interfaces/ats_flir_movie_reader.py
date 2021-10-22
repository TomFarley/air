import os
import numpy as np

"""Set of functions that allow to extract all information I need directly from the .ats file of the FLIR camera.
Fabio Federici
2020-12-11
"""

def hex8_to_int(hex):
	temp = hex[-2:]+hex[4:6]+hex[2:4]+hex[0:2]
	return int(temp,16)

def hex4_to_int(hex):
	temp = hex[-2:]+hex[0:2]
	return int(temp,16)

def hex8_to_float(hex):
	import struct
	temp = hex[-2:]+hex[4:6]+hex[2:4]+hex[0:2]
	return struct.unpack('!f', bytes.fromhex(temp))[0]

def FLIR_record_header_decomposition(header):
	# There is certainly a field for the filter used, but I don't have files with different filters on to compare.
	camera_type = bytearray.fromhex(header[678:724]).decode()

	width = hex4_to_int(header[814:818])
	height = hex4_to_int(header[818:822])

	camera_SN = int(bytearray.fromhex(header[1034:1050]).decode())
	lens = bytearray.fromhex(header[1066:1074]).decode()
	return dict([('camera_type',camera_type),('width',width),('height',height),('camera_SN',camera_SN),('lens',lens)])

def FLIR_frame_header_decomposition(header):
	def return_requested_output(request):
		if request=='time':
			return hex8_to_int(header[16:24])	# ns
		elif request=='frame_counter':
			return hex8_to_int(header[32:40])
		elif request=='DetectorTemp':
			return hex8_to_float(header[64:72])	# K
		elif request=='SensorTemp_0':
			return hex8_to_float(header[72:80])	# K
		elif request=='SensorTemp_1':
			return hex8_to_float(header[80:88])	# K
		elif request=='SensorTemp_2':
			return hex8_to_float(header[88:96])	# K
		elif request=='SensorTemp_3':
			return hex8_to_float(header[96:104])	# K
		elif request=='MasterClock':
			return hex8_to_float(header[104:112])	# unknown
		elif request=='IntegrationTime':
			return hex8_to_float(header[112:120])	# ns
		elif request=='FrameRate':
			return hex8_to_float(header[120:128])	# Hz
		elif request=='ExternalTrigger':
			return int(header[130:132])
		elif request=='PageIndex':
			return hex8_to_int(header[134:142])	# index of the digitiser used
	return return_requested_output



def raw_to_image(raw_digital_level,width,height,digital_level_bytes):
	import textwrap
	pixels = width*height
	# raw_digital_level_splitted = textwrap.wrap(raw_digital_level, 4)
	# iterator=map(hex4_to_int,raw_digital_level_splitted)
	# return np.flip(np.array(list(iterator)).reshape((height,width)),axis=0)
	counts_digital_level = []
	for i in range(pixels):
		counts_digital_level.append(hex4_to_int(raw_digital_level[i*digital_level_bytes:(i+1)*digital_level_bytes]))
	# frame = np.flip(np.array(counts_digital_level).reshape((height,width)),axis=0)
	frame = np.array(counts_digital_level).reshape((height,width))
	return frame

def read_ats_file_header(path, fileats, digital_level_bytes=4, header_marker='4949'):
	data = open(os.path.join(path, fileats), 'rb').read()
	hexdata = data.hex()
	# raw_for_digitizer = b'\x18K\x00\x00zD\x00\x00 A\x00\x00\x00'
	# header_marker = '4949'
	length_header_marker = len(header_marker)
	header_length = 142
	# raw_pointer_after_string = 11 + len(hex_for_digitizer)
	hex_pointer_after_string = 15 + length_header_marker
	header = FLIR_record_header_decomposition(hexdata)
	return header

def ats_to_dict(path, fileats, digital_level_bytes=4, header_marker='4949', n_frames=None):
	"""Read an ats movie file produced by FLIR IR cameras.

	Digitiser_ID: Which of two digitisers is used to read off each frame aquisition. Each digitiser has slightly
	different properties and may need different NUCs to be applied for temperature sensitive measurements.

	Args:
	    path: Location of movie file
	    fileats: Move file name
	    digital_level_bytes: Number of bytes per pixel digital level value
	    header_marker:

	Returns: Dict of movie data and header data

	"""
	data = open(os.path.join(path,fileats),'rb').read()
	hexdata = data.hex()
	# raw_for_digitizer = b'\x18K\x00\x00zD\x00\x00 A\x00\x00\x00'
	# header_marker = '4949'
	length_header_marker = len(header_marker)
	header_length = 142
	# raw_pointer_after_string = 11 + len(hex_for_digitizer)
	hex_pointer_after_string = 15 + length_header_marker
	header = FLIR_record_header_decomposition(hexdata)
	camera = header['camera_type']
	camera_SN = header['camera_SN']
	lens = header['lens']
	width = header['width']
	height = header['height']
	# digital_level_bytes = 4
	data_length = width*height*digital_level_bytes
	digitizer_ID = []
	data = []
	time_of_measurement = []
	frame_counter = []
	DetectorTemp = []
	SensorTemp_0 = []
	last = 0
	import time as tm
	# value = data.find(string_for_digitizer)
	value = hexdata.find(header_marker)
	i = 0
	# last+=value+len(hex_for_digitizer)	# the first one is part of the neader of the whole file
	# value = hexdata[last:].find(hex_for_digitizer)
	while len(hexdata)-last>header_length:
		# start_time = tm.time()
		# print(len(time_of_measurement))
		header = hexdata[last+value+length_header_marker:last+value+header_length+length_header_marker]
		header = FLIR_frame_header_decomposition(header)
		digitizer_ID.append(header('PageIndex'))
		time_of_measurement.append(header('time'))
		frame_counter.append(header('frame_counter'))
		DetectorTemp.append(header('DetectorTemp'))
		SensorTemp_0.append(header('SensorTemp_0'))
		# time_lapsed = tm.time()-start_time
		# print(time_lapsed)
		if (n_frames is None) or (i+1 <= n_frames):
			# To quickly retreive meta data, skip reading frame data
			raw_digital_level = hexdata[last+value-data_length:last+value]
			# time_lapsed = tm.time()-start_time-time_lapsed
			# print(time_lapsed)
			data.append(raw_to_image(raw_digital_level,width,height,digital_level_bytes))
			# time_lapsed = tm.time()-start_time-time_lapsed
			# print(time_lapsed)
		last+=value+header_length+data_length
		if len(time_of_measurement)<=1:	# the spacing between separators seems constant, and take very long, so I do it once
			value = hexdata[last:].find(header_marker)
			IntegrationTime = header('IntegrationTime')
			FrameRate = header('FrameRate')
			ExternalTrigger = header('ExternalTrigger')
		i += 1

		# print(value)
	data = np.array(data)
	digitizer_ID = np.array(digitizer_ID)
	time_of_measurement = np.array(time_of_measurement)
	frame_counter = np.array(frame_counter)
	DetectorTemp = np.array(DetectorTemp)
	SensorTemp_0 = np.array(SensorTemp_0)

	# return data,digitizer_ID,time_of_measurement,IntegrationTime,FrameRate,ExternalTrigger,SensorTemp_0,DetectorTemp,width,height,camera_SN,frame_counter
	movie_data = dict([('data',data),('digitizer_ID',digitizer_ID),('time_of_measurement',time_of_measurement),
				  ('IntegrationTime',IntegrationTime),('FrameRate',FrameRate),('ExternalTrigger',ExternalTrigger),
				('SensorTemp_0',SensorTemp_0),('DetectorTemp',DetectorTemp),
					   ('height',height), ('width',width),  # Swap names here?
					   ('camera', camera), ('camera_SN',camera_SN), ('lens', lens), ('frame_counter',frame_counter)])
	return movie_data

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	# path = '/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018'
	# file_ats = 'irvb_sample-000001.ats'

	path = '/home/tfarley/data/movies/mast_u/FLIR_data-tfarley/'
	file_ats = 'flir_air_sequence_16-04-2021.ats'

	print(f'Reading FLIR ats movie (may be slow): {path}/{file_ats}')
	movie_data = ats_to_dict(path, file_ats)

	print(movie_data)

	plt.figure(f'ats movie {file_ats}')
	plt.imshow(movie_data['data'][10])
	plt.show()