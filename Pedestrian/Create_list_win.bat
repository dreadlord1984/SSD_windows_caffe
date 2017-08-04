@echo off 
REM Author: xyLiu 01/12/2016
REM this file is used to generate trainval.txt, test.txt, which include 
REM the image path and the corresponding annotation path, test_name_size.txt,
REM which inlcudes the name of image (w/o extension) and its corresponding width and height

REM set the root directory of the data, the data should be structured like this 
REM %ROOTDIR%/%DATASET%/Annotations/
REM %ROOTDIR%/%DATASET%/JPEGImages/
REM %ROOTDIR%/%DATASET%/ImageSets/Main
REM and the test, trainval lists should be put in %ROOTDIR%/ImageSets/Main
REM the following lines should be changed accordingly

set ROOTDIR=\\192.168.1.186\PedestrianData
set DATASET=Data_0728
set OUTDATASET=Data_0731
REM the directory where save the result lists.
set DSTDIR=E:\caffe-master_\Pedestrian
REM name of the trainval and test list,  could be changed
set VAL=list\test_min_box
set TRAIN=list\test_min_box
REM the location of get_image_size.exe
set BINDIR=E:\caffe-master_\Build\x64\Release
REM Matlab Dir
set MATLABDIR=C:\Program Files\MATLAB\MATLAB Production Server\R2015a\bin   

REM flag = 1 use get_image_size.exe; flag=0 use matlab
set flag=1

REM no need to change the following lines
set SUBDIR=%DATASET%
set ANNODIR=%DATASET%/Annotations
set JPEGDIR=%DATASET%/JPEGImages
REM set local enable delayed expansion
REM for /f %%i in ("%ROOTDIR%") do (
	REM set PRE=%%~ni
REM )



if exist %DSTDIR%\%OUTDATASET%\val.txt (
	del %DSTDIR%\%OUTDATASET%\val.txt
)
if exist %DSTDIR%\%OUTDATASET%\train.txt (
	del %DSTDIR%\%OUTDATASET%\train.txt
)
if exist %DSTDIR%\%OUTDATASET%\val_name_size.txt (
	del %DSTDIR%\%OUTDATASET%\val_name_size.txt
)
echo start to genrate %TRAIN% list...
for /f "tokens=*" %%i in (%ROOTDIR%\%SUBDIR%\%TRAIN%.txt) do (
	echo %JPEGDIR%/%%i.jpg %ANNODIR%/%%i.xml>> %DSTDIR%/%OUTDATASET%/train.txt	
)
echo start to genrate %VAL% list...
for /f "tokens=*" %%i in (%ROOTDIR%\%SUBDIR%\%VAL%.txt) do (
	echo %JPEGDIR%/%%i.jpg %ANNODIR%/%%i.xml>> %DSTDIR%/%OUTDATASET%/val.txt	
)

echo start to generate %VAL%_name_size list... 

echo use get_image_size.exe to get the image size...
"%BINDIR%"\get_image_size.exe %ROOTDIR% %DSTDIR%\%OUTDATASET%\val.txt %DSTDIR%\%OUTDATASET%\val_name_size.txt

pause