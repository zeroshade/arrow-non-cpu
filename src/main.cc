/* Most of this code is taken from the basic example of libcudf
 * located at https://github.com/rapidsai/cudf/blob/branch-23.06/cpp/examples/basic/src/process_csv.cpp
 *
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "helpers.h"

#include <cudf/aggregation.hpp>
#include <cudf/reduction.hpp>
#include <cudf/groupby.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/table/table.hpp>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

cudf::io::table_with_metadata read_csv(std::string const& file_path)
{
  auto source_info = cudf::io::source_info(file_path);
  auto builder     = cudf::io::csv_reader_options::builder(source_info);
  auto options     = builder.build();
  return cudf::io::read_csv(options);
}

void write_csv(cudf::table_view const& tbl_view, std::string const& file_path)
{
  auto sink_info = cudf::io::sink_info(file_path);
  auto builder   = cudf::io::csv_writer_options::builder(sink_info, tbl_view);
  auto options   = builder.build();
  cudf::io::write_csv(options);
}

std::vector<cudf::groupby::aggregation_request> make_single_aggregation_request(
  std::unique_ptr<cudf::groupby_aggregation>&& agg, cudf::column_view value)
{
  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back(cudf::groupby::aggregation_request());
  requests[0].aggregations.push_back(std::move(agg));
  requests[0].values = value;
  return requests;
}

std::unique_ptr<cudf::table> average_closing_price(cudf::table_view stock_info_table)
{
  // Schema: | Company | Date | Open | High | Low | Close | Volume |
  auto keys = cudf::table_view{{stock_info_table.column(0)}};  // Company
  auto val  = stock_info_table.column(5);                      // Close

  // Compute the average of each company's closing price with entire column
  cudf::groupby::groupby grpby_obj(keys);
  auto requests =
    make_single_aggregation_request(cudf::make_mean_aggregation<cudf::groupby_aggregation>(), val);

  auto agg_results = grpby_obj.aggregate(requests);

  // Assemble the result
  auto result_key = std::move(agg_results.first);
  auto result_val = std::move(agg_results.second[0].results[0]);
  std::vector<cudf::column_view> columns{result_key->get_column(0), *result_val};
  return std::make_unique<cudf::table>(cudf::table_view(columns));
}

/********************************/
struct memory_handler {
  rmm::mr::cuda_memory_resource cuda_mr;
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> mr{&cuda_mr};
};

std::unique_ptr<memory_handler> mh;

#ifdef __cplusplus
extern "C" {
#endif

void initialize_cudf() {
  mh = std::make_unique<memory_handler>();
  rmm::mr::set_current_device_resource(&mh->mr);
}

void get_data_cudf(struct ArrowDeviceArray* output) {
  // Read data
  auto stock_table_with_metadata = read_csv("4stock_5day.csv");

  // Process
  auto result = average_closing_price(*stock_table_with_metadata.tbl);

  to_arrow_device_arr(std::move(result), output);
}

int get_sum(struct ArrowDeviceArray* input, struct ArrowDeviceArray* output) {
  cudaSetDevice(input->device_id);
  
  // for now we're going to assume float32, but you could pass a struct ArrowSchema*
  // to pass the type information alongside the input
  auto ev = reinterpret_cast<cudaEvent_t*>(input->sync_event);
  auto status = cudaStreamWaitEvent(cudf::get_default_stream(), *ev);
  if (status != cudaSuccess) {
    std::cout << cudaGetErrorName(status) << " " << cudaGetErrorString(status) << std::endl;
    return 1;
  }
  
  auto col = cudf::column_view(cudf::data_type(cudf::type_id::FLOAT32),
                               static_cast<cudf::size_type>(input->array.length),
                               input->array.buffers[1],
                               nullptr,
                               0);
  
  auto sumagg = cudf::make_sum_aggregation();
  auto scalar = cudf::reduce(col,
                             *dynamic_cast<cudf::reduce_aggregation*>(sumagg.get()),
                             cudf::data_type(cudf::type_id::FLOAT32));
  
  auto result = cudf::make_column_from_scalar(*scalar, 1);
  to_arrow_device_arr(std::move(result), output);
  status = cudaEventRecord(*reinterpret_cast<cudaEvent_t*>(output->sync_event), cudf::get_default_stream());
  if (status != cudaSuccess) {
    std::cout << cudaGetErrorName(status) << " " << cudaGetErrorString(status) << std::endl;
    return 1;
  }

  input->array.release(&input->array);

  return 0;
}

void cleanup_cudf() {
  mh.reset();
}

#ifdef __cplusplus
}
#endif
