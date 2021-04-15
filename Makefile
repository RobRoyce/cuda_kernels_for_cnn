CPP=g++
NVCC=nvcc
CFLAGS=--std=c++11 -g

MODULE := conv1 conv2 class1 mnist

all: $(MODULE)

HEADERS=

class1: classifier.cu $(HEADERS)
	$(NVCC) $^ $(CFLAGS) -o $@ -DNi=32 -DNn=16
	
mnist: mnist.cu $(HEADERS)
	$(NVCC) $^ $(CFLAGS) -o $@

conv1: convolution.cu $(HEADERS)
	$(NVCC) $^ $(CFLAGS) -o $@ -DNx=224 -DNy=224 -DKx=3  -DKy=3  -DNi=64  -DNn=64        -DTii=32 -DTi=16  -DTnn=32 -DTn=16 -DTx=7 -DTy=7

conv2: convolution.cu $(HEADERS)
	$(NVCC) $^ $(CFLAGS) -o $@ -DNx=14 -DNy=14   -DKx=3  -DKy=3  -DNi=512  -DNn=512      -DTii=32 -DTi=16  -DTnn=32 -DTn=16 -DTx=2 -DTy=2

clean:
	rm -f $(MODULE)