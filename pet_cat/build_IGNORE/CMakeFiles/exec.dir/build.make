# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/shiven/.local/lib/python3.5/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/shiven/.local/lib/python3.5/site-packages/cmake/data/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/shiven/Desktop/pet_cat

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/shiven/Desktop/pet_cat/build

# Include any dependencies generated for this target.
include CMakeFiles/exec.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/exec.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/exec.dir/flags.make

CMakeFiles/exec.dir/src/tut.cpp.o: CMakeFiles/exec.dir/flags.make
CMakeFiles/exec.dir/src/tut.cpp.o: ../src/tut.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shiven/Desktop/pet_cat/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/exec.dir/src/tut.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/exec.dir/src/tut.cpp.o -c /home/shiven/Desktop/pet_cat/src/tut.cpp

CMakeFiles/exec.dir/src/tut.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/exec.dir/src/tut.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shiven/Desktop/pet_cat/src/tut.cpp > CMakeFiles/exec.dir/src/tut.cpp.i

CMakeFiles/exec.dir/src/tut.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/exec.dir/src/tut.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shiven/Desktop/pet_cat/src/tut.cpp -o CMakeFiles/exec.dir/src/tut.cpp.s

# Object files for target exec
exec_OBJECTS = \
"CMakeFiles/exec.dir/src/tut.cpp.o"

# External object files for target exec
exec_EXTERNAL_OBJECTS =

exec: CMakeFiles/exec.dir/src/tut.cpp.o
exec: CMakeFiles/exec.dir/build.make
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_stitching3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_superres3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_videostab3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_aruco3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_bgsegm3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_bioinspired3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_ccalib3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_cvv3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_dpm3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_face3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_fuzzy3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_hdf3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_img_hash3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_line_descriptor3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_optflow3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_reg3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_rgbd3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_saliency3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_stereo3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_structured_light3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_surface_matching3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_tracking3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_xfeatures2d3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_ximgproc3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_xobjdetect3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_xphoto3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_shape3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_photo3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_datasets3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_plot3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_text3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_dnn3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_ml3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_video3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_calib3d3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_features2d3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_highgui3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_videoio3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_viz3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_phase_unwrapping3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_flann3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_imgcodecs3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_objdetect3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_imgproc3.so.3.3.1
exec: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_core3.so.3.3.1
exec: CMakeFiles/exec.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/shiven/Desktop/pet_cat/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable exec"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/exec.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/exec.dir/build: exec

.PHONY : CMakeFiles/exec.dir/build

CMakeFiles/exec.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/exec.dir/cmake_clean.cmake
.PHONY : CMakeFiles/exec.dir/clean

CMakeFiles/exec.dir/depend:
	cd /home/shiven/Desktop/pet_cat/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shiven/Desktop/pet_cat /home/shiven/Desktop/pet_cat /home/shiven/Desktop/pet_cat/build /home/shiven/Desktop/pet_cat/build /home/shiven/Desktop/pet_cat/build/CMakeFiles/exec.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/exec.dir/depend
