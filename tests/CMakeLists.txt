set(test
  test_main.cpp
)

add_executable(app ${test})

target_include_directories(app PUBLIC ../include)

add_library(test_lib STATIC ${test})

target_compile_features(app PUBLIC cxx_std_17)

target_link_libraries(app test_lib)
