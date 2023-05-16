from ffi_headers import ffi, release_callback
import warnings
from time import perf_counter, sleep
import numpy as np
import numba
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

print(np.__version__)
print(numba.__version__)

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

lib = ffi.dlopen("./build/libcudf-demo.so")
lib.initialize_cudf()

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
def single_thread_sum(partial_reduction, sum):
    sum[0] = 0.0
    for element in partial_reduction:
        sum[0] += element


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
    h = ffi.from_buffer(ev.handle)
    out.sync_event = ffi.cast("void*", h)
    
    out.array.length = arr.size
    out.array.null_count = 0
    out.array.offset = 0
    out.array.n_buffers = 2
    out.array.n_children = 0
    buffers = ffi.new("void*[2]")    
    buf = ffi.from_buffer(arr.device_ctypes_pointer)
    buffers[1] = ffi.cast("void*", buf)
    out.array.buffers = buffers
    out.array.private_data = ffi.new_handle(buffers)
    out.array.release = release_callback
    return out

def call_release_devarr(arr):
    arr.array.release(ffi.addressof(arr.array))

class ArrowDevArrayF32:
    def __init__(self, arr):
        self._arr = ffi.gc(arr, call_release_devarr, 0)

    def get_event(self):
        return cuda.driver.Event(cuda.current_context(), self._arr.sync_event)

    @property
    def __cuda_array_interface__(self):
        return {
            'shape': tuple(self._arr.array.length),
            'strides': None,
            # assume float32 for now, could use ArrowSchema to indicate the
            # type properly, but not doing that in this demo
            'typestr': 'f32', 
            # we're gonna manually sync via the event instead!
            'stream': None,
            'data': (int(ffi.cast('uintptr_t', self._arr.array.buffers[1]))),
            'version': 3,            
        }

# Define host array
a = np.ones(10_000_000, dtype=np.float32)
print(f"Old sum: {a.sum():.2f}")

event_ready = cuda.event()

# Pin memory
with cuda.pinned(a):
    # Create a CUDA stream
    stream = cuda.stream()

    # Array copy to device and creation in the device. With Numba, you pass the
    # stream as an additional to API functions.
    dev_a = cuda.to_device(a, stream=stream)
    dev_a_reduce = cuda.device_array((blocks_per_grid,), dtype=dev_a.dtype, stream=stream)
    dev_a_sum = cuda.device_array((1,), dtype=dev_a.dtype, stream=stream)

    # When launching kernels, stream is passed to the kernel launcher ("dispatcher")
    # configuration, and it comes after the block dimension (`threads_per_block`)
    partial_reduce[blocks_per_grid, threads_per_block, stream](dev_a, dev_a_reduce)
    event_ready.record(stream=stream)

    c_dev_a_reduce = to_arrow_device_arr(dev_a_reduce, event_ready)
    output = ffi.new("struct ArrowDeviceArray*")
    st = lib.get_sum(c_dev_a_reduce, output)
    if st != 0:
        raise Exception

    # single_thread_sum[1, 1, stream](dev_a_reduce, dev_a_sum)

    arrowdev = ArrowDevArrayF32(output)
    ev = arrowdev.get_event()
    ev.wait(stream=stream)

    divide_by[blocks_per_grid, threads_per_block, stream](dev_a, dev_a_sum)

    # Array copy to host: like the copy to device, when a stream is passed, the copy
    # is asynchronous. Note: the printed output will probably be nonsensical since
    # the write has not been synchronized yet.
    dev_a.copy_to_host(a, stream=stream)

# Whenever we want to ensure that all operations in a stream are finished from
# the point of view of the host, we call:
stream.synchronize()

print(a)
# After that call, we can be sure that `a` has been overwritten with its
# normalized version
print(f"New sum: {a.sum():.2f}")

lib.cleanup_cudf()