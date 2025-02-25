rm -rf build
mkdir build
cd build
cmake ..
make
./main
#leaks --atExit -- ./main
# ./main_test
