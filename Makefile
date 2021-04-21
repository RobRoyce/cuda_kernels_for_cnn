CPP=g++
NVCC=nvcc
CFLAGS=--std=c++11

MODULE := conv1 conv1b conv2 conv2b class1

all: $(MODULE)

HEADERS=

debug: convolution.cu $(HEADERS)
	$(NVCC) $^ $(CFLAGS) -g -G -o $@ -DNx=224 -DNy=224 -DKx=3  -DKy=3  -DNi=64  -DNn=64        -DTii=32 -DTi=16  -DTnn=32 -DTn=16 -DTx=7 -DTy=7

class1: classifier.cu $(HEADERS)
	$(NVCC) $^ $(CFLAGS) -o $@ -DNi=4096 -DNn=1024

conv1: convolution.cu $(HEADERS)
	$(NVCC) $^ $(CFLAGS) -o $@ -DNx=224 -DNy=224 -DKx=3  -DKy=3  -DNi=64  -DNn=64 -DTii=32 -DTi=16  -DTnn=32 -DTn=16 -DTx=7 -DTy=7 -DBatch=1

conv1b: convolution.cu $(HEADERS)
	$(NVCC) $^ $(CFLAGS) -o $@ -DNx=224 -DNy=224 -DKx=3  -DKy=3  -DNi=64  -DNn=64 -DTii=32 -DTi=16  -DTnn=32 -DTn=16 -DTx=7 -DTy=7 -DBatch=16

conv2: convolution.cu $(HEADERS)
	$(NVCC) $^ $(CFLAGS) -o $@ -DNx=14 -DNy=14 -DKx=3  -DKy=3  -DNi=512  -DNn=512 -DTii=32 -DTi=16  -DTnn=32 -DTn=16 -DTx=2 -DTy=2 -DBatch=1

conv2b: convolution.cu $(HEADERS)
	$(NVCC) $^ $(CFLAGS) -o $@ -DNx=14 -DNy=14 -DKx=3  -DKy=3  -DNi=512  -DNn=512 -DTii=32 -DTi=16  -DTnn=32 -DTn=16 -DTx=2 -DTy=2 -DBatch=16

clean:
	rm -f $(MODULE)
