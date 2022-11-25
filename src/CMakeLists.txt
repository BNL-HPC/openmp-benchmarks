set(test_sources
  catch_main.cpp
  array_init.cu
  array_init.cpp
)

add_executable(main_app main.cpp)
include_directories(main_app PUBLIC ../include)

add_executable(test_app ${test_sources})
target_include_directories(test_app PUBLIC ../include)

