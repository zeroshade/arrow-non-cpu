from ffi_headers import ffi
import numba
import numba.cuda
import numpy as np

lib = ffi.dlopen("./build/libcudf-demo.so")
lib.initialize_cudf()

output = ffi.new("struct ArrowDeviceArray*")
lib.get_data_cudf(output)

print(f"output.device_type: {output.device_type} -> should be 2 for ARROW_DEVICE_CUDA")
assert output.device_type == 2
print(f"output.device_id: {output.device_id}")
numba.cuda.select_device(output.device_id)

# second column is a float64 column, lets use that
# grab the data buffer of the column
ptr = int(ffi.cast('uintptr_t', output.array.children[1].buffers[1]))
# this stuff might be overkill, but it matches how the pointer and object
# are constructed in numba. The better solution would likely be to
# create python wrapping object that implements the `__cuda_array_interface__`
# attribute which would give easy importing through numba. But for now
# this is sufficient to prove this works.
devptr = numba.cuda.driver.get_devptr_for_active_ctx(ptr)
shape, strides, dtype = numba.cuda.prepare_shape_strides_dtype((4,), None, np.float64, order='C')
size = numba.cuda.driver.memory_size_from_info(shape, strides, dtype.itemsize)
data = numba.cuda.driver.MemoryPointer(numba.cuda.current_context(), devptr, size=size, owner=None)
da = numba.cuda.devicearray.DeviceNDArray(shape=shape, strides=strides, dtype=dtype, gpu_data=data, stream=0)
result = da.copy_to_host()
print(result)

# top level release will clean up everything
output.array.release(ffi.addressof(output.array))

lib.cleanup_cudf()