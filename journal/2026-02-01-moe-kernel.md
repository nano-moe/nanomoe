# MoE Kernel implementation

Need to be careful about the precision, as the expert weights, which will be used for merging expert 
outputs, can have different precision (float32 for stability) than the residual stream (typically bfloat16 or lower),
therefore certain in-place operation, such as `mul_`