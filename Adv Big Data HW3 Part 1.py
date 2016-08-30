#Anubha Bhargava
import time
import csv
import numpy as np
import pyopencl as cl
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
mpl.rcParams['savefig.dpi'] = 100

from pylab import *

# Set up platform and the devices on each platform
NAME = 'NVIDIA CUDA'
platforms = cl.get_platforms()
devices = None
for platform in platforms:
    if platform.name == NAME:
        devices = platform.get_devices()

ctx=cl.Context(devices)
queue=cl.CommandQueue(ctx)

print "Multiplication of y=A*B"

# Naive Implementation Kernel

func4= cl.Program(ctx,"""                                       
#pragma OPENCL EXTENSION cl_khr_fp64: enable
__kernel void mat_multiply(__global float* A, __global float* B, __global float* C, unsigned int size3){
        const int i = get_group_id(0);
        const int j = get_group_id(1);
        unsigned int k; 
        float acc=0.0;
        for (k=0; k<size3; k++) {
        
                acc += (A[i*size3 + k] * (B[k*size3 + j]));     
        }
        C[i*size3 + j] = acc;
}
""").build().mat_multiply
func4.set_scalar_arg_dtypes([None, None, None, np.uint32])
      
# Functions

# Get data for pyopencl naive implementation:
def pyopencl_trans(a_buf, b_buf,c_buf,size_2):
        start = time.time()
        func4(queue, (size_2,size_2), None, a_buf, b_buf, c_buf, np.uint32(size_2))
        return time.time()-start
    
def pyopencl_res(a,b,c,size_2):
        a_buf,b_buf,c_buf = mem_buffer(a,b,c)
        t=pyopencl_trans(a_buf,b_buf,c_buf,size_2)

        c=cp_final_data(c,c_buf)
        return t, c

def pyopencl_res_time(size_2, M=4):
        times = []
        a=np.random.random((size_2,size_2)).astype(np.float32)
        b=np.random.random((size_2,size_2)).astype(np.float32)
        c=np.zeros((size_2,size_2)).astype(np.float32)
        a_buf, b_buf, c_buf = mem_buffer(a, b, c)
        for i in xrange(M):
             t=pyopencl_trans(a_buf,b_buf, c_buf, size_2)
             times.append(t)
             c=cp_final_data(c,c_buf)
        return np.average(times)

# General functions
def cp_final_data(C, c_buf):
        cl.enqueue_copy(queue,C,c_buf)
        return C

def mem_buffer(A, B, C):
        mf=cl.mem_flags
        a_buf=cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        b_buf=cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        c_buf=cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)
        init_arr=np.zeros(C.shape).astype(np.float32)
        cl.enqueue_copy(queue,c_buf,init_arr)
        return a_buf, b_buf, c_buf

# Get data for Python implementation:

def python_res(A,B,y):
        start=time.time()
        y=np.dot(A,B)
        t= time.time()-start
        return t,y

def python_res_time(size_2,M=4):
        times = []
        a=np.random.random((size_2,size_2)).astype(np.float32)
        b=np.random.random((size_2,size_2)).astype(np.float32)
        c=np.zeros((size_2,size_2)).astype(np.float32)
        a_buf, b_buf, c_buf = mem_buffer(a, b, c)
        for i in xrange(M):

               t,y=python_res(a,b,c)
               times.append(t)
        return np.average(times)

# Call the functions

size3=32
m=size3
n=size3
p=size3
MAX_PARAM = 50


a=np.random.random((size3,size3)).astype(np.float32)
b=np.random.random((size3,size3)).astype(np.float32)
c=np.zeros((size3,size3)).astype(np.float32)
a_buf, b_buf, c_buf = mem_buffer(a, b, c)
c=cp_final_data(c, c_buf)

python_time,C_py =python_res(a, b, c)
pyopencl_time4, C_cl4=pyopencl_res(a,b,c,size3)

python_times=[]
pyopencl_times=[]

param=np.arange(1,MAX_PARAM,1).astype(np.int32)

# Print the Results

print "\nDimensions\t", "Python Time\t", "Naive Implementation Time\t"

for j in param:
        python_times.append(python_res_time(j*size3,4))
        pyopencl_times.append(pyopencl_res_time(j*size3,4))

for j in param:
        print "(",j*size3, ",",j*size3,")\t", python_times[j-1],"\t", pyopencl_times[j-1], "\t"

plt.clf()
plt.plot(param*size3, python_times, 'ro-',
         param*size3, pyopencl_times, 'b*-')
plt.xlabel('# of Elements in Matrices A,B,C')
plt.ylabel('$t$')
plt.title('Difference in Computation Time for Python and Naive')
plt.legend(('Python', 'Naive A*B'), loc='upper right')
plt.grid(True)
plt.gca().set_xlim((min(param*size3), max(param*size3)))
plt.gca().set_ylim((0, 1.2*max(python_times)))
plt.savefig('Implementations.png')
