/*type==0: Collect; type==1: Distribute*/
int SparsePSAForward_gpu(int type, int kernel_size, THCudaTensor *pool, THCudaTensor *Re, THCudaTensor *atten,
                        THCudaTensor *bottom1, THCudaTensor *bottom2, THCudaTensor *top);
int SparsePSABackward_gpu(int type, int kernel_size, THCudaTensor *bottom1, THCudaTensor *bottom2, THCudaTensor *top, 
    THCudaTensor *bottom1_grad, THCudaTensor *bottom2_grad, THCudaTensor *top_grad);