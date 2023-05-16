#include "abi.h"
#include <cudf/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/utilities/traits.hpp>

struct dispatch_to_arrow_cdata{
    template <typename T, CUDF_ENABLE_IF(not cudf::is_rep_layout_compatible<T>())>
    void operator()(cudf::column_view input_view,
                    cudf::type_id id,
                    struct ArrowArray* out) {
        CUDF_FAIL("unsupported type for to_arrow_cdata");
    }

    template <typename T, CUDF_ENABLE_IF(cudf::is_rep_layout_compatible<T>())>
    void operator()(cudf::column_view input_view,
                    cudf::type_id id,
                    struct ArrowArray* out) {
        *out = (struct ArrowArray) {
            .length = input_view.size(),
            .null_count = input_view.null_count(),
            .offset = 0,
            .n_buffers = 2,
            .n_children = 0,
            .buffers = (const void**)(malloc(sizeof(void*) * 2)),
            .children = nullptr,
            .dictionary = nullptr,
            .release = 
                [](struct ArrowArray* arr) {
                    free(arr->buffers);                    
                },
            .private_data = nullptr,
        };

        out->buffers[0] = input_view.null_mask();
        out->buffers[1] = input_view.data<T>();
    }
};

template <>
void dispatch_to_arrow_cdata::operator()<cudf::string_view>(
    cudf::column_view input_view, cudf::type_id id, struct ArrowArray* out) {
    
    *out = (struct ArrowArray) {
        .length = input_view.size(),
        .null_count = input_view.null_count(),
        .offset = 0,
        .n_buffers = 3,
        .n_children = 0,
        .buffers = (const void**)(malloc(sizeof(void*)*3)),
        .children = nullptr,
        .dictionary = nullptr,
        .release = 
            [](struct ArrowArray* arr) {
                free(arr->buffers);
            },
        .private_data = nullptr,
    };
    out->buffers[0] = input_view.null_mask();
    out->buffers[1] = input_view.child(0).data<int32_t>();
    out->buffers[2] = input_view.child(1).data<const char>();
}

struct dev_ctx {
    std::unique_ptr<cudf::column> col;
    cudaEvent_t ev;
};

void to_arrow_device_arr(std::unique_ptr<cudf::column> col, struct ArrowDeviceArray* out) {
    memset(out, 0, sizeof(struct ArrowDeviceArray));

    int device;
    cudaGetDevice(&device);
    
    out->device_id = static_cast<int64_t>(device);
    out->device_type = ARROW_DEVICE_CUDA;
    auto view = col->view();
    cudf::type_dispatcher(view.type(), dispatch_to_arrow_cdata{}, view, view.type().id(), &out->array);
    out->array.release = [](struct ArrowArray* arr) {
        free(arr->buffers);
        auto self_ctx = reinterpret_cast<dev_ctx*>(arr->private_data);
        cudaEventDestroy(self_ctx->ev);
        delete self_ctx;
    };

    auto* ctx = new dev_ctx;    
    cudaEventCreate(&ctx->ev);
    ctx->col = std::move(col);

    out->sync_event = reinterpret_cast<void*>(ctx->ev);
    out->array.private_data = reinterpret_cast<void*>(ctx);
}

void to_arrow_device_arr(std::unique_ptr<cudf::table> tbl, struct ArrowDeviceArray* out) {
    memset(out, 0, sizeof(struct ArrowDeviceArray));
    
    int device;
    cudaGetDevice(&device);

    *out = (struct ArrowDeviceArray) {
        .array = (struct ArrowArray) {
            .length = tbl->num_rows(),
            .null_count = 0,
            .offset = 0,
            .n_buffers = 1,
            .n_children = tbl->num_columns(),
            .buffers = (const void**)(malloc(sizeof(void*))),
            .children = (struct ArrowArray**)(malloc(sizeof(struct ArrowArray*)*tbl->num_columns())),
            .dictionary = nullptr,
            .release =
                [](struct ArrowArray* arr) {
                    auto* self_tbl = reinterpret_cast<std::unique_ptr<cudf::table>*>(arr->private_data);
                    free(arr->buffers);
                    for (int i = 0; i < arr->n_children; ++i) {
                        arr->children[i]->release(arr->children[i]);
                        free(arr->children[i]);
                    }
                    free(arr->children);
                    delete self_tbl;
                },            
        },
        .device_id = static_cast<int64_t>(device),
        .device_type = ARROW_DEVICE_CUDA,        
    };    

    out->array.buffers[0] = nullptr;
    for (int i = 0; i < tbl->num_columns(); ++i) {
        out->array.children[i] = (struct ArrowArray*)(malloc(sizeof(struct ArrowArray)));
        memset(out->array.children[i], 0, sizeof(struct ArrowArray));

        auto col = tbl->get_column(i).view();
        cudf::type_dispatcher(col.type(), dispatch_to_arrow_cdata{}, col, col.type().id(), out->array.children[i]);
    }    

    out->array.private_data = reinterpret_cast<void*>(new std::unique_ptr<cudf::table>(std::move(tbl)));
}