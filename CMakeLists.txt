cmake_minimum_required (VERSION 3.17)
project (TIPL VERSION 0 LANGUAGES CXX)
configure_file (tipl.pc.in tipl.pc @ONLY)


# Install pkg-config file
install (FILES ${CMAKE_CURRENT_BINARY_DIR}/tipl.pc DESTINATION share/pkgconfig)

# INSTALL header files
install (FILES def.hpp mt.hpp DESTINATION include/ )
install (FILES tipl.hpp DESTINATION include/TIPL)
foreach(dir "cuda" "filter" "io" "ml" "morphology" "numerical" "reg" "segmentation" "utility" "vis") 
 file (GLOB_RECURSE headers ${dir}/*.hpp)
 install (FILES ${headers} DESTINATION include/${dir}/)
endforeach()

# Make an interface Library Modern CMAKE target
add_library(tipl INTERFACE)

find_package(Threads)

# Add this on to the list of includes for users
target_include_directories(tipl INTERFACE $<INSTALL_INTERFACE:include>)

if( Threads_FOUND ) 
  target_link_libraries(tipl INTERFACE Threads::Threads )
endif()

# Set some properties -- technically for portability
# we should switch off GNU targets, but for now let us
# Leave them on for 'anonymous-structs`
set_target_properties(tipl PROPERTIES 
                      CXX_STANDARD 17)



# Create the library export
install(TARGETS tipl EXPORT TIPLTargets
  INCLUDES DESTINATION include
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(EXPORT TIPLTargets
  FILE TIPLTargets.cmake
  NAMESPACE TIPL::
  DESTINATION lib/cmake/TIPL)

add_library(TIPL::tipl ALIAS tipl)


include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  TIPLConfigVersion.cmake
  VERSION "${PACKAGE_VERSION}"
  COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(TIPLConfig.cmake.in TIPLConfig.cmake
  INSTALL_DESTINATION lib/cmake/TIPL
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/TIPLConfigVersion.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/TIPLConfig.cmake 
          DESTINATION lib/cmake/TIPL)


