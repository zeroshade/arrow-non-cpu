# arrow-non-cpu Demo

Toy example using the proposed Arrow Non-CPU enhancements for the C-Data
API.

This uses python to call an extern "C" function exposed from 
libcudf-demo.so which will utilize libcudf to parse the `4stock_5day.csv`
into a `cudf::table` and then computes the average closing price for
each ticker in the csv file. This result is returned from the function
via an `ArrowDeviceArray` structure as struct column with two fields.

The second column is then wrapped using `numba.cuda` into a 
`DeviceNDArray` without performing any copies and leaving the data on the
GPU the whole time. We then confirm that it worked properly by using numba
to perform a `copy_to_host` on the data and print out the result.

## Requirements

Building this requires the following:

* libcudf
* python 3.10+
* numba
* cmake 3.23+

### Conda

libcudf can be installed with conda/mamba from the `rapidsai` channel.

```shell
conda install -c rapidsai -c conda-forge -c nvidia cudf python=3.10 cudatoolkit
```

## Building

Set up to use CMake:

```shell
cmake -S . -B build/
cmake --build build/ --config Release
```

This will build `libcudf-demo.so` in the `build/` subdirectory. After that
you can run `demo.py` as long as `numba` and `numpy` are installed. 