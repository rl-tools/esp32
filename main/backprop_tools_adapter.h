#ifdef __cplusplus
extern "C"
#endif
void backprop_tools_run(float* output_mem);

#ifdef __cplusplus
extern "C"
#endif
void backprop_tools_run_single_sample(float* output_mem);

#ifdef __cplusplus
extern "C"
#endif
float backprop_tools_run_single_sample_check_output(float* output_mem);

