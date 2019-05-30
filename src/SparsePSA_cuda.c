#include <THC/THC.h>
#include <math.h>
#include "SparsePSA_cuda.h"
#include "cuda/SparsePSA_cuda_kernel.h"
extern THCState *state;
int SparsePSAForward_gpu(int type, int kernel_size, THCudaTensor *pool, THCudaTensor *Re, THCudaTensor *atten,
                        THCudaTensor *bottom1, THCudaTensor *bottom2, THCudaTensor *top)
{    
    float *bottom1_data = THCudaTensor_data(state, bottom1);
    float *bottom2_data = THCudaTensor_data(state, bottom2);       
    float *top_data = THCudaTensor_data(state, top);
    //temp parameter
    float *pool_data = THCudaTensor_data(state, pool);
    float *Re_data = THCudaTensor_data(state, Re);
    float *atten_data = THCudaTensor_data(state, atten);

    int num =  THCudaTensor_size(state, bottom1, 0);
    int channels = THCudaTensor_size(state, bottom1, 1);
    int height = THCudaTensor_size(state, bottom1, 2);
    int width = THCudaTensor_size(state, bottom1, 3);        
    cudaStream_t stream = THCState_getCurrentStream(state);
    SparsePSAForward_gpu_kernel(type, kernel_size, num, channels, height, width, 
        pool_data, Re_data, atten_data,
        bottom1_data, bottom2_data, top_data, stream);
    return -1;    
}


int SparsePSABackward_gpu(int type, int kernel_size, THCudaTensor *bottom1, THCudaTensor *bottom2, THCudaTensor *top, 
    THCudaTensor *bottom1_grad, THCudaTensor *bottom2_grad, THCudaTensor *top_grad)/*psatyp=0 collect, psatype=1 distribute*/
{    
    float *bottom1_data = THCudaTensor_data(state, bottom1);
    float *bottom2_data = THCudaTensor_data(state, bottom1);//n*channels*height*width
    float *top_data = THCudaTensor_data(state, top);

    float *bottom1_diff = THCudaTensor_data(state, bottom1_grad);
    float *bottom2_diff = THCudaTensor_data(state, bottom2_grad);
    float *top_diff = THCudaTensor_data(state, top_grad);

    int num =  THCudaTensor_size(state, bottom1, 0);
    int channels = THCudaTensor_size(state, bottom1, 1);
    int height = THCudaTensor_size(state, bottom1, 2);
    int width = THCudaTensor_size(state, bottom1, 3);    
    cudaStream_t stream = THCState_getCurrentStream(state);
    SparsePSABackward_gpu_kernel(type, kernel_size, num, channels,  height, width, 
        bottom1_data, bottom2_data, top_data, 
        bottom1_diff, bottom2_diff, top_diff, stream);
    return -1;    
}