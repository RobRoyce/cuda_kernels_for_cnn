CPP=g++
NVCC=nvcc
CFLAGS=--std=c++11 -g

MODULE := conv1 conv1b conv2 conv2b class1 class2 mnist diannao_conv1 diannao_conv2 diannao_class1 diannao_class2

all: $(MODULE)

HEADERS=

debug: convolution.cu $(HEADERS)
	$(NVCC) $^ $(CFLAGS) -g -G -o $@ -DNx=224 -DNy=224 -DKx=3  -DKy=3  -DNi=64  -DNn=64        -DTii=32 -DTi=16  -DTnn=32 -DTn=16 -DTx=7 -DTy=7

class1: classifier.cu $(HEADERS)
	$(NVCC) $^ $(CFLAGS) -o $@ -DNi=25088 -DNn=4096   -DTii=512 -DTi=64     -DTnn=32  -DTn=16

class2: classifier.cu $(HEADERS)
	$(NVCC) $^ $(CFLAGS) -o $@ -DNi=4096 -DNn=1024    -DTii=32 -DTi=32      -DTnn=32  -DTn=16
	
mnist: mnist.cu $(HEADERS)
	$(NVCC) $^ $(CFLAGS) -o $@

conv1: convolution.cu $(HEADERS)
	$(NVCC) $^ $(CFLAGS) -o $@ -DNx=224 -DNy=224 -DKx=3  -DKy=3  -DNi=64  -DNn=64 -DTii=32 -DTi=16  -DTnn=32 -DTn=16 -DTx=7 -DTy=7 -DBatch=1

conv1b: convolution.cu $(HEADERS)
	$(NVCC) $^ $(CFLAGS) -o $@ -DNx=224 -DNy=224 -DKx=3  -DKy=3  -DNi=64  -DNn=64 -DTii=32 -DTi=16  -DTnn=32 -DTn=16 -DTx=7 -DTy=7 -DBatch=16

conv2: convolution.cu $(HEADERS)
	$(NVCC) $^ $(CFLAGS) -o $@ -DNx=14 -DNy=14 -DKx=3  -DKy=3  -DNi=512  -DNn=512 -DTii=32 -DTi=16  -DTnn=32 -DTn=16 -DTx=2 -DTy=2 -DBatch=1

conv2b: convolution.cu $(HEADERS)
	$(NVCC) $^ $(CFLAGS) -o $@ -DNx=14 -DNy=14 -DKx=3  -DKy=3  -DNi=512  -DNn=512 -DTii=32 -DTi=16  -DTnn=32 -DTn=16 -DTx=2 -DTy=2 -DBatch=16

diannao_conv1: diannao_conv.cu $(HEADERS)
	$(NVCC) $^ $(CFLAGS) -o $@ -DNx=224 -DNy=224 -DKx=3  -DKy=3  -DNi=64  -DNn=64 -DTii=32 -DTi=16  -DTnn=32 -DTn=16 -DTx=7 -DTy=7 -DBatch=1

diannao_conv2: diannao_conv.cu $(HEADERS)
	$(NVCC) $^ $(CFLAGS) -o $@ -DNx=14 -DNy=14 -DKx=3  -DKy=3  -DNi=512  -DNn=512 -DTii=32 -DTi=16  -DTnn=32 -DTn=16 -DTx=2 -DTy=2 -DBatch=1

diannao_class1: diannao_class.cu $(HEADERS)
	$(NVCC) $^ $(CFLAGS) -o $@ -DNi=25088 -DNn=4096   -DTii=512 -DTi=64     -DTnn=32  -DTn=16

diannao_class2: diannao_class.cu $(HEADERS)
	$(NVCC) $^ $(CFLAGS) -o $@ -DNi=4096 -DNn=1024    -DTii=32 -DTi=32      -DTnn=32  -DTn=16

clean:
	rm -f $(MODULE)
