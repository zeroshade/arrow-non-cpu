from ffi_headers import ffi, release_callback
from ctypes import c_voidp, addressof
from time import perf_counter, sleep
import numpy as np
import numba
from numba import cuda

print(np.__version__)
print(numba.__version__)

lib = ffi.dlopen("./build/libcudf-demo.so")
lib.initialize_cudf()

# sample numba cuda code adapted from
# https://towardsdatascience.com/cuda-by-numba-examples-7652412af1ee

threads_per_block = 256
blocks_per_grid = 32 * 40

@cuda.jit
def partial_reduce(array, partial_reduction):
    i_start = cuda.grid(1)
    threads_per_grid = cuda.blockDim.x * cuda.gridDim.x
    s_thread = 0.0
    for i_arr in range(i_start, array.size, threads_per_grid):
        s_thread += array[i_arr]

    s_block = cuda.shared.array((threads_per_block,), numba.float32)
    tid = cuda.threadIdx.x
    s_block[tid] = s_thread
    cuda.syncthreads()

    i = cuda.blockDim.x // 2
    while (i > 0):
        if (tid < i):
            s_block[tid] += s_block[tid + i]
        cuda.syncthreads()
        i //= 2

    if tid == 0:
        partial_reduction[cuda.blockIdx.x] = s_block[0]

@cuda.jit
def divide_by(array, val_array):
    i_start = cuda.grid(1)
    threads_per_grid = cuda.gridsize(1)
    for i in range(i_start, array.size, threads_per_grid):
        array[i] /= val_array[0]

@cuda.require_context
def to_arrow_device_arr(arr, ev):
    out = ffi.new("struct ArrowDeviceArray*")
    out.device_id = cuda.get_current_device().id
    out.device_type = 2 # ARROW_DEVICE_CUDA
    # gotta play some games since numba uses ctypes and I decided
    # to use ffi to call the shared lib.
    # the void* we want is actually the address of ev.handle as ev.handle
    # is the cudaEvent_t itself.
    out.sync_event = ffi.cast('void*', ffi.cast('uintptr_t', addressof(ev.handle)))
    out.array.length = arr.size
    out.array.null_count = 0
    out.array.offset = 0
    out.array.n_buffers = 2
    out.array.n_children = 0
    out.array.buffers = ffi.new("void*[2]")
    out.array.buffers[1] = ffi.cast("void*", ffi.cast('uintptr_t', arr.device_ctypes_pointer.value))
    out.array.private_data = ffi.new_handle(out.array.buffers)
    out.array.release = release_callback
    return out

class ArrowDevArrayF32:
    """Simple little class to wrap an ArrowDeviceArray"""
    def __init__(self):
        self._arr = ffi.new("struct ArrowDeviceArray*")
        # okay technically to "properly" handle the memory
        # this we should be calling arr.array.release
        # when this gets cleaned up, but I'm not going to deal with
        # that for this toy example

    def get_event(self):
        # the handle for the event in numba expects to be the cudaEvent_t
        # itself, but we get a void* which is really a cudaEvent_t*
        # so we need to play some games between ffi and ctypes in order
        # to properly have a valid address here to use as a handle.
        h = c_voidp.from_address(int(ffi.cast('uintptr_t', self._arr.sync_event)))
        return cuda.driver.Event(cuda.current_context(), h)

    @property
    def c_arr(self):
        return self._arr

    @property
    def __cuda_array_interface__(self):
        # use the CAI to provide the pointers and info directly to
        # numba so it can use the passed array
        return {
            # it's a single dimension array so just a tuple with one value
            'shape': (self._arr.array.length,),
            'strides': None,
            # assume float32 for now, could use ArrowSchema to indicate the
            # type properly, but not doing that in this demo
            'typestr': 'float32',
            # we're gonna manually sync via the event instead!
            'stream': None,
            'data': (int(ffi.cast('uintptr_t', self._arr.array.buffers[1])), True),
            'version': 3,
        }

# Define host array
a = np.ones(10_000_000, dtype=np.float32)
print(f"Old sum: {a.sum():.2f}")

# this event will denote when the partial reduce is complete
# so that the C++ side can wait on it before computing the sum
event_ready = cuda.event()

# Pin memory
with cuda.pinned(a):
    # Create a CUDA stream
    stream = cuda.stream()

    # Array copy to device and creation in the device. With Numba, you pass the
    # stream as an additional to API functions.
    dev_a = cuda.to_device(a, stream=stream)
    dev_a_reduce = cuda.device_array((blocks_per_grid,), dtype=dev_a.dtype, stream=stream)

    # When launching kernels, stream is passed to the kernel launcher ("dispatcher")
    # configuration, and it comes after the block dimension (`threads_per_block`)
    partial_reduce[blocks_per_grid, threads_per_block, stream](dev_a, dev_a_reduce)
    # add a record on the stream for this event so we can sync on it
    event_ready.record(stream=stream)

    # create the struct ArrowDeviceArray from the numba.DeviceNDArray
    c_dev_a_reduce = to_arrow_device_arr(dev_a_reduce, event_ready)
    # create output struct and call the C++ function which will
    # use libcudf to compute the sum of the column and return it as a
    # single element array
    output = ArrowDevArrayF32()
    st = lib.get_sum(c_dev_a_reduce, output.c_arr)
    if st != 0:
        raise Exception

    # get the event so we can tell the stream to wait until the c++ side
    # completes the sum to sync between the streams
    ev = output.get_event()
    ev.wait(stream=stream)

    dev_a_sum = cuda.as_cuda_array(output, sync=False)
    divide_by[blocks_per_grid, threads_per_block, stream](dev_a, dev_a_sum)

    # Array copy to host: like the copy to device, when a stream is passed, the copy
    # is asynchronous. Note: the printed output will probably be nonsensical since
    # the write has not been synchronized yet.
    dev_a.copy_to_host(a, stream=stream)

# Whenever we want to ensure that all operations in a stream are finished from
# the point of view of the host, we call:
stream.synchronize()

# After that call, we can be sure that `a` has been overwritten with its
# normalized version
print(f"New sum: {a.sum():.2f}")
# should print New sum: 1.00
lib.cleanup_cudf()