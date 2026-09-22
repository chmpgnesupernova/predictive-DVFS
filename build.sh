  g++ -std=c++17 main.cpp -o vision_test \
  -I/usr/include/opencv4 \
  $(pkg-config --cflags --libs opencv4) \
  -lonnxruntime -lpthread
