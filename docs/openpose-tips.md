---
title: OpenPose Installation Tips and Troubleshooting
author: 
- Sebastian Barrios 
date: Fri Dec 03 2021
---
# OpenPose Installation Tips and Troubleshooting

**Following this [Guide](https://medium.com/@alok.gandhi2002/build-openpose-with-without-gpu-support-for-macos-catalina-10-15-6-8fb936c9ab05)**

## Fix cblas.h error when building when using homebrew:

1. Open `./CMakeList.txt` and change the line:
```bash
if (APPLE)
    include_directories("/opt/homebrew/opt/openblas/include")
endif (APPLE)
```

## Fix "Could NOT find vecLib" when building:

1. Find your `vecLib.h` file. In my case it was in: `/Library/Developer/CommandLineTools/SDKs/MacOSX11.3.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers/vecLib.h`
2. Edit in `./build/caffe/src/openpose_lib-build/CMakeCache.txt` the line:
```bash
\\vecLib include directory
vecLib_INCLUDE_DIR:PATH=/Library/Developer/CommandLineTools/SDKs/MacOSX11.3.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers/
```

## Fix error  "Undefined symbols for architecture arm64" when building:
```bash
[ 39%] Linking CXX shared library libopenpose.dylib
Undefined symbols for architecture arm64:
  "caffe::Net<float>::CopyTrainedLayersFrom(std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >)", referenced from:
      op::NetCaffe::initializationOnThread() in netCaffe.cpp.o
ld: symbol(s) not found for architecture arm64
clang: error: linker command failed with exit code 1 (use -v to see invocation)
make[2]: *** [src/openpose/libopenpose.1.7.0.dylib] Error 1
make[1]: *** [src/openpose/CMakeFiles/openpose.dir/all] Error 2
```
1. Enable flag `BUILD_PYTHON` 