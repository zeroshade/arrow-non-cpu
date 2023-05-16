#pragma once

#include "abi.h"
#include <cudf/table/table.hpp>

void to_arrow_device_arr(std::unique_ptr<cudf::column> tbl, struct ArrowDeviceArray* out);
void to_arrow_device_arr(std::unique_ptr<cudf::table> tbl, struct ArrowDeviceArray* out);
