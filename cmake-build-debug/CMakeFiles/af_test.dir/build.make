# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/clion/152/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /snap/clion/152/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/bohdan/CLionProjects/tensortorchAf

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/bohdan/CLionProjects/tensortorchAf/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/af_test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/af_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/af_test.dir/flags.make

CMakeFiles/af_test.dir/src/main.cpp.o: CMakeFiles/af_test.dir/flags.make
CMakeFiles/af_test.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bohdan/CLionProjects/tensortorchAf/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/af_test.dir/src/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/af_test.dir/src/main.cpp.o -c /home/bohdan/CLionProjects/tensortorchAf/src/main.cpp

CMakeFiles/af_test.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/af_test.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bohdan/CLionProjects/tensortorchAf/src/main.cpp > CMakeFiles/af_test.dir/src/main.cpp.i

CMakeFiles/af_test.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/af_test.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bohdan/CLionProjects/tensortorchAf/src/main.cpp -o CMakeFiles/af_test.dir/src/main.cpp.s

CMakeFiles/af_test.dir/src/dense.cpp.o: CMakeFiles/af_test.dir/flags.make
CMakeFiles/af_test.dir/src/dense.cpp.o: ../src/dense.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bohdan/CLionProjects/tensortorchAf/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/af_test.dir/src/dense.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/af_test.dir/src/dense.cpp.o -c /home/bohdan/CLionProjects/tensortorchAf/src/dense.cpp

CMakeFiles/af_test.dir/src/dense.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/af_test.dir/src/dense.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bohdan/CLionProjects/tensortorchAf/src/dense.cpp > CMakeFiles/af_test.dir/src/dense.cpp.i

CMakeFiles/af_test.dir/src/dense.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/af_test.dir/src/dense.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bohdan/CLionProjects/tensortorchAf/src/dense.cpp -o CMakeFiles/af_test.dir/src/dense.cpp.s

CMakeFiles/af_test.dir/src/model.cpp.o: CMakeFiles/af_test.dir/flags.make
CMakeFiles/af_test.dir/src/model.cpp.o: ../src/model.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bohdan/CLionProjects/tensortorchAf/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/af_test.dir/src/model.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/af_test.dir/src/model.cpp.o -c /home/bohdan/CLionProjects/tensortorchAf/src/model.cpp

CMakeFiles/af_test.dir/src/model.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/af_test.dir/src/model.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bohdan/CLionProjects/tensortorchAf/src/model.cpp > CMakeFiles/af_test.dir/src/model.cpp.i

CMakeFiles/af_test.dir/src/model.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/af_test.dir/src/model.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bohdan/CLionProjects/tensortorchAf/src/model.cpp -o CMakeFiles/af_test.dir/src/model.cpp.s

CMakeFiles/af_test.dir/src/BGD.cpp.o: CMakeFiles/af_test.dir/flags.make
CMakeFiles/af_test.dir/src/BGD.cpp.o: ../src/BGD.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bohdan/CLionProjects/tensortorchAf/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/af_test.dir/src/BGD.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/af_test.dir/src/BGD.cpp.o -c /home/bohdan/CLionProjects/tensortorchAf/src/BGD.cpp

CMakeFiles/af_test.dir/src/BGD.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/af_test.dir/src/BGD.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bohdan/CLionProjects/tensortorchAf/src/BGD.cpp > CMakeFiles/af_test.dir/src/BGD.cpp.i

CMakeFiles/af_test.dir/src/BGD.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/af_test.dir/src/BGD.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bohdan/CLionProjects/tensortorchAf/src/BGD.cpp -o CMakeFiles/af_test.dir/src/BGD.cpp.s

# Object files for target af_test
af_test_OBJECTS = \
"CMakeFiles/af_test.dir/src/main.cpp.o" \
"CMakeFiles/af_test.dir/src/dense.cpp.o" \
"CMakeFiles/af_test.dir/src/model.cpp.o" \
"CMakeFiles/af_test.dir/src/BGD.cpp.o"

# External object files for target af_test
af_test_EXTERNAL_OBJECTS =

af_test: CMakeFiles/af_test.dir/src/main.cpp.o
af_test: CMakeFiles/af_test.dir/src/dense.cpp.o
af_test: CMakeFiles/af_test.dir/src/model.cpp.o
af_test: CMakeFiles/af_test.dir/src/BGD.cpp.o
af_test: CMakeFiles/af_test.dir/build.make
af_test: /usr/lib/libafopencl.so.3.8.0
af_test: CMakeFiles/af_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/bohdan/CLionProjects/tensortorchAf/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable af_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/af_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/af_test.dir/build: af_test

.PHONY : CMakeFiles/af_test.dir/build

CMakeFiles/af_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/af_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/af_test.dir/clean

CMakeFiles/af_test.dir/depend:
	cd /home/bohdan/CLionProjects/tensortorchAf/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/bohdan/CLionProjects/tensortorchAf /home/bohdan/CLionProjects/tensortorchAf /home/bohdan/CLionProjects/tensortorchAf/cmake-build-debug /home/bohdan/CLionProjects/tensortorchAf/cmake-build-debug /home/bohdan/CLionProjects/tensortorchAf/cmake-build-debug/CMakeFiles/af_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/af_test.dir/depend

