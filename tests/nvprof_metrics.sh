nvprof -m all --csv --log-file metrics/metrics_conv1.csv ./conv1
nvprof -m all --csv --log-file metrics/metrics_conv1b.csv ./conv1b
nvprof -m all --csv --log-file metrics/metrics_conv2.csv ./conv2
nvprof -m all --csv --log-file metrics/metrics_conv2b.csv ./conv2b
nvprof -m all --csv --log-file metrics/metrics_class1.csv ./class1
nvprof -m all --csv --log-file metrics/metrics_class1b.csv ./class1b
nvprof -m all --csv --log-file metrics/metrics_class2.csv ./class2
nvprof -m all --csv --log-file metrics/metrics_class2b.csv ./class2b
