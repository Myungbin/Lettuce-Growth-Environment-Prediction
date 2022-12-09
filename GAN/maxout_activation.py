import torch
from torch.autograd import Function, Variable


class Maxout(Function):
    
    @staticmethod
    def forward(ctx, input):
        x = input
        max_out=4    
        kernels = x.shape[1] 
        feature_maps = int(kernels / max_out)
        out_shape = (x.shape[0], feature_maps, max_out, x.shape[2], x.shape[3])
        x= x.view(out_shape)
        y, indices = torch.max(x[:, :, :], 2)
        ctx.save_for_backward(input)
        ctx.indices=indices
        ctx.max_out=max_out
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input1,indices,max_out= ctx.saved_variables[0],Variable(ctx.indices),ctx.max_out
        input=input1.clone()
        for i in range(max_out):
            a0=indices==i
            input[:,i:input.data.shape[1]:max_out]=a0.float()*grad_output
      

        return input