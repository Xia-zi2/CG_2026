project(3_PoissonImageEditing)
file(GLOB source
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.h" 
  "${CMAKE_CURRENT_SOURCE_DIR}/shapes/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shapes/*.h" 
  "${CMAKE_CURRENT_SOURCE_DIR}/Poisson/*.cpp"
 "${CMAKE_CURRENT_SOURCE_DIR}/Poisson/*.h"
)
add_executable(${PROJECT_NAME} ${source} "Poisson/PoissonRegion.h" "Poisson/PoissonRegion.cpp" "Poisson/PoissonSolver.h" "Poisson/PoissonSolver.cpp" "Poisson/PoissonBasic.h" "Poisson/PoissonBasic.cpp" "shapes/freehand.h" "shapes/freehand.cpp"  )
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(${PROJECT_NAME} PROPERTIES 
  DEBUG_POSTFIX "_d"
  RUNTIME_OUTPUT_DIRECTORY "${BINARY_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${LIBRARY_DIR}"
  ARCHIVE_OUTPUT_DIRECTORY "${LIBRARY_DIR}") 
target_link_libraries(${PROJECT_NAME} PUBLIC common) 
target_compile_definitions(${PROJECT_NAME} PRIVATE -DDATA_PATH="${FRAMEWORK2D_DIR}/../Homeworks/3_poisson_image_editing/data")