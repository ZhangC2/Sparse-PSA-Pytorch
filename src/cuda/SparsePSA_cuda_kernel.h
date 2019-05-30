#ifndef _SparsePSA_KERNEL
#define _SparsePSA_KERNEL

#ifdef __cplusplus
extern "C" {
#endif
float *pool1, *Re1, *atten1;
void SparsePSAForward_gpu_kernel(int type, int kernel, int batch, int full_feature_channels, int full_feature_h, int full_feature_w, 
                                float *pool, float *Re, float *atten,
                                float *bottom1, float *bottom2, float *top, cudaStream_t stream);
void SparsePSABackward_gpu_kernel(int type, int kernel, int batch, int full_feature_channels, int full_feature_h, int full_feature_w, 
                        float *bottom1, float *bottom2, float *top, 
                        float *bottom1_diff, float *bottom2_diff, float *top_diff, cudaStream_t stream); 

#ifdef __cplusplus
}

#endif

#endif