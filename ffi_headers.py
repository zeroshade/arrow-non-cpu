import cffi

# cffi doesn't support handing it #include directives
# so i'll just reproduce the struct definitions here.
funcs = """
    struct ArrowSchema {
        // Array type description
        const char* format;
        const char* name;
        const char* metadata;
        int64_t flags;
        int64_t n_children;
        struct ArrowSchema** children;
        struct ArrowSchema* dictionary;

        // Release callback
        void (*release)(struct ArrowSchema*);
        // Opaque producer-specific data
        void* private_data;
    };

    struct ArrowArray {
        // Array data description
        int64_t length;
        int64_t null_count;
        int64_t offset;
        int64_t n_buffers;
        int64_t n_children;
        const void** buffers;
        struct ArrowArray** children;
        struct ArrowArray* dictionary;

        // Release callback
        void (*release)(struct ArrowArray*);
        // Opaque producer-specific data
        void* private_data;
    };

    typedef int32_t ArrowDeviceType;

    struct ArrowDeviceArray {
        struct ArrowArray array;
        int64_t device_id;  
        ArrowDeviceType device_type;
        void* sync_event;
        int64_t reserved[3];
    };

    void initialize_cudf();
    void cleanup_cudf();
    void get_data_cudf(struct ArrowDeviceArray*);
    int get_sum(struct ArrowDeviceArray* input, struct ArrowDeviceArray* output);
"""


ffi = cffi.FFI()
ffi.cdef(funcs)

@ffi.callback("void(struct ArrowArray*)")
def release_callback(arr):    
    # do some cleanup if you like
    pass