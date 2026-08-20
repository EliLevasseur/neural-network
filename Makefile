.PHONY: run

run:
	mkdir -p build
	clear
	g++ -std=c++17 -g3 -O0 main.cpp src/*.cpp -o build/nnet
	./build/nnet



test:
	mkdir -p build
	clear
	g++ -std=c++17 -g3 -O0 -DTEST_MODE main.cpp src/*.cpp -o build/nnet
	build/nnet
