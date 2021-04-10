CPP=g++
NVCC=nvcc
CFLAGS=--std=c++11

MODULE := conv1 conv2

all: $(MODULE)

HEADERS=

conv1: convolution.cu $(HEADERS)
	$(CPP) $^ $(CFLAGS) -static -o $@ -DNx=224 -DNy=224 -DKx=3  -DKy=3  -DNi=64  -DNn=64        -DTii=32 -DTi=16  -DTnn=32 -DTn=16 -DTx=7 -DTy=7

conv2: convolution.cu $(HEADERS)
	$(CPP) $^ $(CFLAGS) -static -o $@ -DNx=14 -DNy=14   -DKx=3  -DKy=3  -DNi=512  -DNn=512      -DTii=32 -DTi=16  -DTnn=32 -DTn=16 -DTx=2 -DTy=2

clean:
	rm -f $(MODULE)