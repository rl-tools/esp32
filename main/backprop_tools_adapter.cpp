
#define LAYER_IN_C_DISABLE_DYNAMIC_MEMORY_ALLOCATION
#include "backprop_tools_adapter.h"

#include <layer_in_c/operations/esp32.h>
#include <layer_in_c/nn/layers/dense/operations_esp32/dsp.h>
#include <layer_in_c/nn/layers/dense/operations_esp32/opt.h>
#include <layer_in_c/nn_models/mlp/operations_generic.h>
#include "data/test_layer_in_c_nn_models_mlp_persist_code.h"
#include "data/test_layer_in_c_nn_models_mlp_evaluation.h"


namespace lic = layer_in_c;

using DEV_SPEC = lic::devices::DefaultESP32Specification<lic::devices::esp32::Hardware::C3>;
using DEVICE = lic::devices::esp32::DSP<DEV_SPEC>;
DEVICE device;
using TI = typename mlp_1::SPEC::TI;
using DTYPE = typename mlp_1::SPEC::T;

constexpr TI BATCH_SIZE = decltype(input::container)::ROWS;
DTYPE input_layer_output_memory[BATCH_SIZE* mlp_1::SPEC::OUTPUT_DIM];
lic::MatrixDynamic<lic::matrix::Specification<DTYPE, TI, BATCH_SIZE, mlp_1::SPEC::HIDDEN_DIM, lic::matrix::layouts::RowMajorAlignment<TI, 1>>> input_layer_output = {(DTYPE*)input_layer_output_memory};
DTYPE buffer_tick_memory[BATCH_SIZE * mlp_1::SPEC::HIDDEN_DIM];
DTYPE buffer_tock_memory[BATCH_SIZE * mlp_1::SPEC::HIDDEN_DIM];

// void backprop_tools_run(float* output_mem){
//     lic::MatrixDynamic<lic::matrix::Specification<DTYPE, TI, BATCH_SIZE, mlp_1::SPEC::HIDDEN_DIM, mlp_1::SPEC::MEMORY_LAYOUT>> buffer_tick = {(DTYPE*)buffer_tick_memory};
//     lic::MatrixDynamic<lic::matrix::Specification<DTYPE, TI, BATCH_SIZE, mlp_1::SPEC::HIDDEN_DIM, mlp_1::SPEC::MEMORY_LAYOUT>> buffer_tock = {(DTYPE*)buffer_tock_memory};

//     decltype(mlp_1::mlp)::template Buffers<BATCH_SIZE> buffers = {buffer_tick, buffer_tock};
//     lic::MatrixDynamic<lic::matrix::Specification<DTYPE, TI, BATCH_SIZE, mlp_1::SPEC::OUTPUT_DIM, lic::matrix::layouts::RowMajorAlignment<TI, 1>>> output = {(DTYPE*)output_mem};
//     lic::evaluate(device, mlp_1::mlp, input::container, output, buffers);
// }
void backprop_tools_run_single_sample(float* output_mem){
    static_assert(BATCH_SIZE >= 1);
    lic::MatrixDynamic<lic::matrix::Specification<DTYPE, TI, 1, mlp_1::SPEC::HIDDEN_DIM, mlp_1::SPEC::MEMORY_LAYOUT>> buffer_tick = {(DTYPE*)buffer_tick_memory};
    lic::MatrixDynamic<lic::matrix::Specification<DTYPE, TI, 1, mlp_1::SPEC::HIDDEN_DIM, mlp_1::SPEC::MEMORY_LAYOUT>> buffer_tock = {(DTYPE*)buffer_tock_memory};

    decltype(mlp_1::mlp)::template Buffers<1> buffers = {buffer_tick, buffer_tock};
    lic::MatrixDynamic<lic::matrix::Specification<DTYPE, TI, 1, mlp_1::SPEC::OUTPUT_DIM, lic::matrix::layouts::RowMajorAlignment<TI, 1>>> output = {(DTYPE*)output_mem};
    auto input_sample = lic::view(device, input::container, lic::matrix::ViewSpec<1, mlp_1::SPEC::INPUT_DIM>{}, 0, 0);
    lic::evaluate(device, mlp_1::mlp, input_sample, output, buffers);
}
float backprop_tools_run_single_sample_check_output(float* output_mem){
    static_assert(BATCH_SIZE >= 1);
    lic::MatrixDynamic<lic::matrix::Specification<DTYPE, TI, 1, mlp_1::SPEC::HIDDEN_DIM, mlp_1::SPEC::MEMORY_LAYOUT>> buffer_tick = {(DTYPE*)buffer_tick_memory};
    lic::MatrixDynamic<lic::matrix::Specification<DTYPE, TI, 1, mlp_1::SPEC::HIDDEN_DIM, mlp_1::SPEC::MEMORY_LAYOUT>> buffer_tock = {(DTYPE*)buffer_tock_memory};

    decltype(mlp_1::mlp)::template Buffers<1> buffers = {buffer_tick, buffer_tock};
    lic::MatrixDynamic<lic::matrix::Specification<DTYPE, TI, 1, mlp_1::SPEC::OUTPUT_DIM, lic::matrix::layouts::RowMajorAlignment<TI, 1>>> output = {(DTYPE*)output_mem};
    auto input_sample = lic::view(device, input::container, lic::matrix::ViewSpec<1, mlp_1::SPEC::INPUT_DIM>{}, 0, 0);
    lic::evaluate(device, mlp_1::mlp, input_sample, output, buffers);
    auto abs_diff = lic::abs_diff(device, output, lic::view(device, expected_output::container, lic::matrix::ViewSpec<1, mlp_1::SPEC::OUTPUT_DIM>{}, 0, 0));
    return abs_diff;
}