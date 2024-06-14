// #define RL_TOOLS_DISABLE_DYNAMIC_MEMORY_ALLOCATION
#include <rl_tools/rl_tools.h>
#include <rl_tools/operations/esp32.h>
#include <rl_tools/utils/generic/typing.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_esp32/dsp.h>
#include <rl_tools/nn/layers/dense/operations_esp32/opt.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include "model.h"

#include "rl_tools_adapter.h"


namespace rlt = rl_tools;

using DEV_SPEC_S3 = rlt::devices::DefaultESP32Specification<rlt::devices::esp32::Hardware::S3>;
using DEV_SPEC_DEFAULT = rlt::devices::DefaultESP32Specification<rlt::devices::esp32::Hardware::DEFAULT>;

using DEVICE_S3 = rlt::devices::esp32::DSP<DEV_SPEC_S3>;
using DEVICE_DEFAULT = rlt::devices::esp32::OPT<DEV_SPEC_DEFAULT>;
using DEVICE = DEVICE_DEFAULT; // adjust this to reflect your device
DEVICE device;
using TI = typename DEVICE::index_t;
using T = typename decltype(rl_tools_export::model::module)::T;

constexpr TI BATCH_SIZE = decltype(rl_tools_export::input::container)::ROWS;
constexpr TI INPUT_DIM = decltype(rl_tools_export::model::module)::INPUT_DIM;
constexpr TI OUTPUT_DIM = decltype(rl_tools_export::model::module)::OUTPUT_DIM;

decltype(rl_tools_export::model::module)::Buffer<BATCH_SIZE, rlt::MatrixStaticTag> buffer;
decltype(rlt::random::default_engine(device.random)) rng;

void rl_tools_run_single_sample(float* output_mem){
    static_assert(BATCH_SIZE >= 1);
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, BATCH_SIZE, OUTPUT_DIM, rlt::matrix::layouts::RowMajorAlignment<TI, 1>>> output = {(T*)output_mem};
    rlt::evaluate(device, rl_tools_export::model::module, rl_tools_export::input::container, output, buffer, rng);
}
float rl_tools_run_single_sample_check_output(float* output_mem){
    rl_tools_run_single_sample(output_mem);
    T abs_diff = 0;
    for(TI i = 0; i < OUTPUT_DIM; i++){
        T diff = get(rl_tools_export::output::container, 0, i) - output_mem[i];
        abs_diff += fabs(diff);
        output_mem[i] = diff;
    }
    return abs_diff;
}
