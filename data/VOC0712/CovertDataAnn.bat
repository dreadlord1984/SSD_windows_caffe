@echo off
REM Function: create the leveldb/lmdb inputs
REM N.B the data should be stored as the sturcture 

set DBTYPE=leveldb

REM The root directory which contains the images and annotations
set ROOTDIR=D:/yhzheng/Project/VOC/VOCdevkit/
REM set	ROOTDIR2 = D:/yhzheng/Project/SSD/caffe-windows-ssd/caffe-master_/data/VOC0712

REM A file with LabelMap protobuf message
set MAPFILE=D:/yhzheng/Project/SSD/caffe-windows-ssd/caffe-master_/data/VOC0712/labelmap_voc.prototxt

REM the directory which includes the convert_annoset.exe
set BINDIR=D:/yhzheng/Project/SSD/caffe-windows-ssd/caffe-master_/Build/x64/Release

REM The directory to store the link of the database files
set OUTDIR=D:/yhzheng/Project/SSD/caffe-windows-ssd/caffe-master_/data/VOC0712

REM resize the image to specified dimension. 
set RESIZE_HEIGHT=300		REM 0 - keep unchanged
set RESIZE_WIDTH=300		REM 0 - keep unchanged

REM in most case, no need to change belows
REM resize the image to one of them (i.e., used in faster rcnn), in ssd, just keep them zeros
set MIN_DIM=0
set MAX_DIM=0

REM the label type is to provide label format
set LABELTYPE=xml

REM Annotation Type (detection or classification)
set ANNOTYPE="detection"

set	TRAINVAL="trainval"
set TEST="test"
set SHUFFLE=true
set GRAY=false
set CHECK_SIZE=false
set CHECK_LABEL=true

set	encode_type="jpg"
set encoded=true

set TRAIN_DATA_LIST=D:/yhzheng/Project/SSD/caffe-windows-ssd/caffe-master_/data/VOC0712/trainval.txt
set VAL_DATA_LIST=D:/yhzheng/Project/SSD/caffe-windows-ssd/caffe-master_/data/VOC0712/test.txt

if not exist %TRAIN_DATA_LIST% (
   echo "Error: %TRAIN_DATA_LIST% does not exist"
   goto :eof
)
if not exist %VAL_DATA_LIST% (
   echo "Error: %VAL_DATA_LIST% does not exist"
   goto :eof
)
echo "create train leveldb(lmdb)..."
SET GLOG_logtostderr = 1

%BINDIR%/convert_annoset.exe %ROOTDIR% %TRAIN_DATA_LIST% %OUTDIR%/%TRAINVAL%_%DBTYPE% --encoded=%encoded% --encode_type=%encode_type% --anno_type=%ANNOTYPE% --label_map_file=%MAPFILE% --min_dim=%MIN_DIM% --max_dim=%MAX_DIM% --resize_width=%RESIZE_WIDTH% --resize_height=%RESIZE_HEIGHT% --check_label=%CHECK_LABEL% --shuffle=%SHUFFLE% --gray=%GRAY% --backend=%DBTYPE% 

SET GLOG_logtostderr = 1
%BINDIR%/convert_annoset.exe %ROOTDIR% %VAL_DATA_LIST% %OUTDIR%/%TEST%_%DBTYPE% --encoded=%encoded% --encode_type=%encode_type% --anno_type=%ANNOTYPE% --label_map_file=%MAPFILE% --min_dim=%MIN_DIM% --max_dim=%MAX_DIM% --resize_width=%RESIZE_WIDTH% --resize_height=%RESIZE_HEIGHT% --check_label=%CHECK_LABEL% --shuffle=%SHUFFLE% --gray=%GRAY% --backend=%DBTYPE%