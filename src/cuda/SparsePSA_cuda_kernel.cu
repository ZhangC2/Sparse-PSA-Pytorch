#include <stdio.h>
#include <math.h>
#include <float.h>
#include "cblas.h"
#include "SparsePSA_cuda_kernel.h"
#define CUDA_1D_KERNEL_LOOP(i, n)\
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;\
       i += blockDim.x * gridDim.x)


/******************************************ForwardFunction*******************************/
//DownSample
__global__ void DownSampleForward_gpu(int nthreads, int full_feature_channels, int full_feature_h, int full_feature_w, 
                                    int down_feature_h, int down_feature_w, int kernel, float *bottom1, float *pool1){//pool1全为0                                 
    CUDA_1D_KERNEL_LOOP(index, nthreads){        
        int c = index %full_feature_channels;
        int n = index /full_feature_channels;
        for(int h=0; h<down_feature_h; h++ ){
            for(int w=0; w<down_feature_w; w++){           
                int h_index=min(h*kernel + kernel/2,full_feature_h-1);
                int w_index=min(w*kernel + kernel/2,full_feature_w-1);     
                pool1[(n*full_feature_channels + c)*down_feature_h*down_feature_w + h*down_feature_w + w]=//!!!channle
                bottom1[(n*full_feature_channels + c)*full_feature_h*full_feature_w + h_index*full_feature_w + w_index];
            }
        }
    }
}
//UpSample
__global__ void UpSampleForward_gpu(int nthreads, int full_feature_channels, int full_feature_h, int full_feature_w,
                                int down_feature_h, int down_feature_w, int kernel, float *Re1, float *top){//top为全0 
    CUDA_1D_KERNEL_LOOP(index, nthreads){
        int c = index %full_feature_channels;
        int n = index /full_feature_channels;
        for(int h=0; h<down_feature_h; h++ ){
            for(int w=0; w<down_feature_w; w++){   
                int h_index=min(h*kernel + kernel/2,full_feature_h-1);
                int w_index=min(w*kernel + kernel/2,full_feature_w-1);    
                top[(n*full_feature_channels + c)*full_feature_h*full_feature_w + h_index*full_feature_w + w_index]=
                Re1[(n*full_feature_channels + c)*down_feature_h*down_feature_w + h*down_feature_h + w];
            }
        }
    }
}
//Distribute
__global__ void DistributeForward_gpu(int nthreads, int down_feature_h, int down_feature_w, float *bottom2, float *atten1){
    int feature_channels = down_feature_h*down_feature_w;
    CUDA_1D_KERNEL_LOOP(index, nthreads){//channel loop // index??? nthreads???
        int w = index % down_feature_w;
        int h = (index / down_feature_w) % down_feature_h;//location in map, h&w choose channel
        int n = index /down_feature_h/down_feature_w;
        for(int hindx=0; h<down_feature_h; h++ ){
            for(int windx=0; w<down_feature_w; w++){                                
                atten1[(n*feature_channels + (h*down_feature_w + w))*down_feature_h*down_feature_w + hindx*down_feature_w +windx]=
                bottom2[(n*feature_channels + hindx*down_feature_w +windx)*down_feature_h*down_feature_w + h*down_feature_w + w];
            }
        }
    }
}
//main function
void SparsePSAForward_gpu_kernel(int type, int kernel, int batch, int full_feature_channels, int full_feature_h, int full_feature_w, 
                                float *pool, float *Re, float *atten,
                                float *bottom1, float *bottom2, float *top, cudaStream_t stream){    
    int down_feature_h=(full_feature_h-(kernel-1))/kernel + 1;
    int down_feature_w=(full_feature_w-(kernel-1))/kernel + 1;                                
    pool1 = pool;
    Re1 = Re;
    atten1 = atten;
    int nthreads_DownUp = batch*full_feature_channels;
    int nthreads_DistriCollect = batch*down_feature_h*down_feature_w;
    int kThreadsPerBlock = 1024;      
    float normalization_factor_ = float(down_feature_h * down_feature_w);   
    //bottom1 -> pool1
    DownSampleForward_gpu<<<(nthreads_DownUp + kThreadsPerBlock-1)/kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>
    (nthreads_DownUp, full_feature_channels, full_feature_h, full_feature_w, down_feature_h, down_feature_w, kernel, bottom1, pool1);//output pool1    
    //bottom2 -> atten1
    switch(type){//0:collect 1:distribute
        case 0:
        atten1 = bottom2;
        case 1:
        DistributeForward_gpu<<<(nthreads_DistriCollect + kThreadsPerBlock-1)/kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>
        (nthreads_DistriCollect, down_feature_h, down_feature_w, bottom2, atten1);
    }
    //pool1&atten1 -> Re1
    for(int n = 0; n < batch; n++) {        
        float *this_pool1 = pool1 + n*full_feature_channels*down_feature_h*down_feature_w;
        float *this_atten1 = atten1 + n*(down_feature_h*down_feature_w)*(down_feature_h*down_feature_w);
        float *this_Re1 = Re1 + n*full_feature_channels*down_feature_h*down_feature_w;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, full_feature_channels, down_feature_h*down_feature_w, down_feature_h*down_feature_w,
            float(1.0/normalization_factor_), this_pool1, down_feature_h*down_feature_w, 
            this_atten1, down_feature_h*down_feature_w,
            float(0), this_Re1, down_feature_h*down_feature_w); 
    }
    //Re1 -> top
    UpSampleForward_gpu<<<(nthreads_DownUp + kThreadsPerBlock-1)/kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>
    (nthreads_DownUp, full_feature_channels, full_feature_h,full_feature_w ,down_feature_h, down_feature_w, kernel, Re1, top);//output top
}



/**************************************Backward Functioon*****************************/
//DownSample
__global__ void DownSampleBackward_gpu (int nthreads, int full_feature_channels, int full_feature_h, int full_feature_w, 
                                 int down_feature_h, int down_feature_w, int kernel, float *pool1_diff, float *bottom1_diff){
    CUDA_1D_KERNEL_LOOP(index, nthreads){
        int c = index %full_feature_channels;
        int n = index /full_feature_channels;
        for(int h=0; h<down_feature_h; h++ ){
            for(int w=0; w<down_feature_w; w++){        
                int h_index=min(h*kernel + kernel/2,full_feature_h-1);
                int w_index=min(w*kernel + kernel/2,full_feature_w-1);                         
                bottom1_diff[(n*full_feature_channels+c)*full_feature_h*full_feature_w + h_index*full_feature_h + w_index]=
                pool1_diff[(n*full_feature_channels+c)*down_feature_h*down_feature_w + h*down_feature_h + w];
            }
        }
    }
}
//Upsample
__global__ void UpSampleBackward_gpu(int nthreads, int full_feature_channels, int full_feature_h, int full_feature_w,
                                 int down_feature_h, int down_feature_w, int kernel, float *top_diff, float *Re1_diff){//top为全0     
    CUDA_1D_KERNEL_LOOP(index, nthreads){
        int c = index %full_feature_channels;
        int n = index /full_feature_channels;
        for(int h=0; h<down_feature_h; h++ ){
            for(int w=0; w<down_feature_w; w++){
                int h_index=min(h*kernel + kernel/2,full_feature_h-1);
                int w_index=min(w*kernel + kernel/2,full_feature_w-1);   
                Re1_diff[(n*full_feature_channels+c)*down_feature_h*down_feature_w + h*down_feature_h + w] =
                top_diff[(n*full_feature_channels+c)*full_feature_h*full_feature_w + h_index*full_feature_h + w_index];
            }
        }
    }
}
//Distribute
__global__ void DistributeBackward_gpu(int nthreads, int down_feature_h, int down_feature_w, float *atten1_diff, float *bottom2_diff){
    int feature_channels = down_feature_h*down_feature_w;
    CUDA_1D_KERNEL_LOOP(index, nthreads){//channel loop // index??? nthreads???
        int w = index % down_feature_w;
        int h = (index / down_feature_w) % down_feature_h;//location in map, h&w choose channel
        int n = index /down_feature_h/down_feature_w;
        for(int hindx=0; h<down_feature_h; h++ ){
            for(int windx=0; w<down_feature_w; w++){                         
                bottom2_diff[(n*feature_channels + hindx*down_feature_w +windx)*down_feature_h*down_feature_w + h*down_feature_w + w]=
                atten1_diff[(n*feature_channels + (h*down_feature_w + w))*down_feature_h*down_feature_w + hindx*down_feature_w +windx];
            }
        }
    }
}
//main function
void SparsePSABackward_gpu_kernel(int type, int kernel, int batch, int full_feature_channels, int full_feature_h, int full_feature_w, 
                        float *bottom1, float *bottom2, float *top, 
                        float *bottom1_diff, float *bottom2_diff, float *top_diff, cudaStream_t stream){        
    int down_feature_h=(full_feature_h-(kernel-1))/kernel + 1;
    int down_feature_w=(full_feature_w-(kernel-1))/kernel + 1;
    float *pool1_diff, *Re1_diff, *atten1_diff;
    cudaMallocManaged((void**)&pool1_diff, sizeof(float)*batch*full_feature_channels*down_feature_h*down_feature_w);
    cudaMallocManaged((void**)&Re1_diff, sizeof(float)*batch*full_feature_channels*down_feature_h*down_feature_w);
    cudaMallocManaged((void**)&atten1_diff, sizeof(float)*batch*down_feature_h*down_feature_w*down_feature_h*down_feature_w);
    cudaMemset(pool1,0,sizeof(pool1_diff));
    cudaMemset(Re1,0,sizeof(Re1_diff));
    cudaMemset(atten1,0,sizeof(atten1_diff));
    int nthreads_DownUp = batch*full_feature_channels;
    int nthreads_DistriCollect = batch*down_feature_h*down_feature_w;
    int kThreadsPerBlock = 1024;      
    float normalization_factor_ = float(down_feature_h * down_feature_w); 
    //BP top_diff -> Re1_diff 
    UpSampleBackward_gpu<<<(nthreads_DownUp + kThreadsPerBlock-1)/kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>
    (nthreads_DownUp, full_feature_channels, full_feature_h, full_feature_w, down_feature_h, down_feature_w, kernel, top_diff, Re1_diff);
    //BP Re1_diff -> atten1_diff
    for (int n=0; n<batch; n++){
        float *this_Re1_diff= Re1_diff + n*full_feature_channels*down_feature_h*down_feature_w;
        float *this_pool1= pool1 + n*full_feature_channels*full_feature_h*full_feature_w;
        float *this_atten1_diff= atten1_diff + n*down_feature_h*down_feature_w*down_feature_h*down_feature_w;
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, down_feature_h*down_feature_w, down_feature_h*down_feature_w, full_feature_channels,
            float(1.0/normalization_factor_), this_pool1, down_feature_h*down_feature_w, 
            this_Re1_diff, down_feature_h*down_feature_w,
            float(0), this_atten1_diff, down_feature_h*down_feature_w);
    }  
    //BP Re1_diff -> pool1_diff
    for (int n=0; n<batch; n++){
        float *this_Re1_diff= Re1_diff + n*full_feature_channels*down_feature_h*down_feature_w;
        float *this_atten1= atten1 + n*down_feature_h*down_feature_w*down_feature_h*down_feature_w;
        float *this_pool1_diff= pool1_diff + n*full_feature_channels*down_feature_h*down_feature_w;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, full_feature_channels, down_feature_h * down_feature_w, down_feature_h*down_feature_w,
            float(1.0/normalization_factor_), this_Re1_diff, down_feature_h * down_feature_w, 
            this_atten1, down_feature_h * down_feature_w,
            float(0), this_pool1_diff, down_feature_h * down_feature_w);         
    }
    //BP pool1_diff -> bottom1_diff
    DownSampleBackward_gpu<<<(nthreads_DownUp + kThreadsPerBlock-1)/kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>
    (nthreads_DownUp, full_feature_channels, full_feature_h, full_feature_w, down_feature_h,down_feature_w, kernel, pool1_diff,bottom1_diff);
    //BP atten1_diff -> bottom2_diff
    switch(type){//0:collect 1:distribute
        case 0:
            bottom2_diff =atten1_diff;
        case 1:
            DistributeBackward_gpu<<<(nthreads_DistriCollect+kThreadsPerBlock-1)/kThreadsPerBlock, kThreadsPerBlock,0, stream>>>
            (nthreads_DistriCollect, down_feature_h, down_feature_w, atten1_diff, bottom2_diff);
    }  
    cudaFree(pool1_diff);
    cudaFree(Re1_diff);
    cudaFree(atten1_diff);
}
