.PHONY: run

run:
	mkdir -p build
	clear
	g++ main.cpp src/* -o build/nnet
	./build/nnet

test:
	mkdir -p build
	clear
	g++ -DTEST_MODE main.cpp src/* -o build/nnet
	./build/nnet
