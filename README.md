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
you can run `demo.py` and/or `events.py` as long as `numba` and `numpy`
are installed. 

## Examples

`demo.py` is a very simple case that uses libcudf to read a csv file into
a `cudf::table`, performs an aggregation, and then returns the table as 
an `ArrowDeviceArray` so that numba can import the device pointers. It 
then copies the result to the host via numba to confirm it got the
expected data. This is based on the [`basic`](https://github.com/rapidsai/cudf/tree/branch-23.06/cpp/examples/basic) example from libcudf.

`events.py` uses numba to perform a partial reduce of a device array,
then pass the device array to C++ using `ArrowDeviceArray` so it can
compute the sum of the reduced array (without leaving the GPU) using
libcudf. The result is then returned for numba to perform a division
on every element by the sum. Events are used to sync between the 
operations in numba and libcudf which will be on different streams.
This is based on the "Stream Semantics in Numba CUDA" tutorial located
at https://towardsdatascience.com/cuda-by-numba-examples-7652412af1ee