
# Create a usage to indicate the needed variables, also create a snippet that
# checks the variables, if there is no station, it does not work. If there are
# no dates, then jumps automatically to Step 5.

# ...


# Put the uncompressed hld files in
#	/media/externalDisk/gate/system/devices/TRB3/data/daqData/rawData/dat
# Collect the asci files in
#	/media/externalDisk/gate/system/devices/TRB3/data/daqData/asci

station=$1 # Station number: 1, 2, 3 or 4

base_directory = /home/cayetano/DATAFLOW_v3/STATIONS/MINGO0$station/ZERO_STAGE

hld_input_directory = /media/externalDisk/gate/system/devices/TRB3/data/daqData/rawData/dat
asci_output_directory = /media/externalDisk/gate/system/devices/TRB3/data/daqData/asci

first_stage_raw_directory = /home/cayetano/DATAFLOW_v3/STATIONS/MINGO0$station/FIRST_STAGE/EVENT_DATA/RAW



compressed_directory = $base_directory/COMPRESSED_HLDS
uncompressed_directory = $base_directory/UNCOMPRESSED_HLDS

processed_directory = $base_directory/ASCII

mkdir $compressed_directory
mkdir $uncompressed_directory
mkdir $processed_directory

# Step 1. Take a data range in YYMMDD, convert it to YYDDDHHMMSS, DDD is day of year

start = $2
end = $3



start_DOY = ...
end_DOY = ...


# Step 2. Collect all the compressed hlds in that range from:
#	rpcuser@backuplip:local/experiments/MINGOS/MINGO0$station
# And bring them to the COMPRESSED_HLDS directory
# The files are called mi0$stationYYDDDHHMMSS.hld*




# Step 3. Uncompress them, erase the compressed


# tar -xvfz all files of COMPRESSED_HLDS into UNCOMPRESSED_HLDS

mv ...

rm $compressed_directory/*tar.gz



# Step 4. Move the uncompressed files to /media/externalDisk/gate/system/devices/TRB3/data/daqData/rawData/dat

mv $uncompressed_directory/* hld_input_directory


# Step 5. Execute the unpacking

export RPCSYSTEM=mingo0$station;export RPCRUNMODE=oneRun;/home/cayetano/gate/bin/unpack.sh


# Step 6. Move the ascii files to ASCII

mv asci_output_directory $processed_directory


# Step 7. Copy the ASCII files into the First stage raw files directory

cp $processed_directory/* first_stage_raw_directory
