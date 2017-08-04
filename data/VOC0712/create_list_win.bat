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

set ROOTDIR=D:\yhzheng\Project\VOC\VOCdevkit
set DATASET=VOC2007
REM the directory where save the result lists.
set DSTDIR=D:\yhzheng\Project\SSD\caffe-windows-ssd\caffe-master_\data\VOC0712
REM name of the trainval and test list,  could be changed
set TEST=test
set TRAINVAL=trainval
REM the location of get_image_size.exe
set BINDIR=D:\yhzheng\Project\SSD\caffe-windows-ssd\caffe-master_\Build\x64\Release
REM Matlab Dir
set MATLABDIR=C:\Program Files\MATLAB\MATLAB Production Server\R2015a\bin

REM flag = 1 use get_image_size.exe; flag=0 use matlab
set flag=1

REM no need to change the following lines
set SUBDIR=%DATASET%/ImageSets/Main
set ANNODIR=%DATASET%/Annotations
set JPEGDIR=%DATASET%/JPEGImages
REM setlocal enabledelayedexpansion
REM for /f %%i in ("%ROOTDIR%") do (
	REM set PRE=%%~ni
REM )

if exist %DSTDIR%\%TEST%.txt (
	del %DSTDIR%\%TEST%.txt
)
if exist %DSTDIR%\%TRAINVAL%.txt (
	del %DSTDIR%\%TRAINVAL%.txt
)
if exist %DSTDIR%\%TEST%_name_size.txt (
	del %DSTDIR%\%TEST%_name_size.txt
)
echo start to genrate %TRAINVAL% list...
for /f %%i in (%ROOTDIR%\%SUBDIR%\%TRAINVAL%.txt) do (
	echo %JPEGDIR%/%%i.jpg %ANNODIR%/%%i.xml >> %DSTDIR%\trainval.txt	
)
echo start to generate %TEST% list... 
for /f %%i in (%ROOTDIR%\%SUBDIR%\%TEST%.txt) do (
	echo %JPEGDIR%/%%i.jpg %ANNODIR%/%%i.xml >> %DSTDIR%\test.txt	
)
echo start to generate %TEST%_name_size list... 
if "%flag%"=="0" (
	echo use Matlab function to get the image size...
	"%MATLABDIR%"\matlab.exe -nojvm -nosplash -nodesktop -nodisplay -r "get_image_size('%ROOTDIR%','%DSTDIR%/test.txt','%DSTDIR%\%TEST%_name_size.txt')"
) else (
	echo use get_image_size.exe to get the image size...
	"%BINDIR%"\get_image_size.exe %ROOTDIR% %DSTDIR%\test.txt %DSTDIR%\%TEST%_name_size.txt
)