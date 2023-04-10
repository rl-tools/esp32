#include <stdio.h>
#include "backprop_tools_adapter.h"
#include <stdbool.h>
#include "esp_timer.h"


#define BACKPROP_TOOLS_OUTPUT_DIM 4
#define BACKPROP_TOOLS_BATCH_SIZE 1
float output_mem[BACKPROP_TOOLS_BATCH_SIZE * BACKPROP_TOOLS_OUTPUT_DIM];
void app_main(void)
{
    backprop_tools_run_single_sample((float*)output_mem);
    for(int batch_i = 0; batch_i < BACKPROP_TOOLS_BATCH_SIZE; batch_i++){
        for(int col_i = 0; col_i < BACKPROP_TOOLS_OUTPUT_DIM; col_i++){
            printf("%+5.5f ", output_mem[batch_i * BACKPROP_TOOLS_OUTPUT_DIM + col_i]);
        }
        printf("\n");
    }
    int num_runs = 100;

    while(true){
        printf("Starting %d runs\n", num_runs);
        int64_t start = esp_timer_get_time();
        for(int i = 0; i < num_runs; i++){
            backprop_tools_run_single_sample((float*)output_mem);
        }
        int64_t end = esp_timer_get_time();
        printf("Time taken: %lld us\n", end - start);
        printf("Frequency: %f Hz\n", num_runs / ((end - start) / 1000000.0));
    }
}