# CUDA Kernels for Convolutional Neural Networks (CNNs)

This project is based on an assignment for UCLA's graduate-level course -- Current Topics in Computer Science: System Design/Architecture: Learning Machines. 

The goal is to design and implement convolution and classification kernels in CUDA that improve on CPU and GPU baselines (see [DianNao](https://dl.acm.org/doi/10.1145/2541940.2541967), and [Baidu's DeepBench](https://github.com/baidu-research/DeepBench), which leverages cuDNN acceleration).

Our kernels improve on DianNao by three orders of magnitude, but fall short of cuDNN by roughly the same.
