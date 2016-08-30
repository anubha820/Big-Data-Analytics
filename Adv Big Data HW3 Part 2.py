#Anubha Bhargava

import csv
import time
import pyopencl as cl
import pyopencl.array
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
mpl.rcParams['savefig.dpi'] = 100


#Choose the PyOpenCL platform
NAME = 'NVIDIA CUDA'
platforms = cl.get_platforms()
devs = None
for platform in platforms:
    if platform.name == NAME:
        devs = platform.get_devices()

#Command Queue
ctx = cl.Context(devs[0:1])
queue = cl.CommandQueue(ctx)
print 'OUTPUT: \n'

def mult_py(a, b, c):
    start = time.time()
    c[:, :] = np.dot(a, b)
    return time.time()-start

## KERNELS

# Transpose matrix algorithm:
kernel= """                                     
#pragma OPENCL EXTENSION cl_khr_fp64: enable

__kernel void mat_transpose(__global float* d, __global float *d_trans, unsigned int row_size, unsigned int col_size) {
        unsigned int i = get_global_id(1);
        unsigned int j = get_global_id(0);

        d_trans[j*row_size + i]= d[i*col_size +j];
}
"""                                             #KERNEL

# Naive OpenCL algorithm:
naive_opencl = cl.Program(ctx, """
#include <pyopencl-complex.h>
__kernel void func(__global const cfloat_t* a, __global const cfloat_t* b, 
                   __global cfloat_t* c,
                   const unsigned int P, const unsigned int Q, const unsigned int R) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);
    unsigned int n;

    c[i*R+j] = cfloat_fromreal(0.0);
    for (n=0; n<Q; n++)
        c[i*R+j] = cfloat_add(c[i*R+j], cfloat_mul(a[i*Q+n], b[n*R+j]));
}
""").build().func
naive_opencl.set_scalar_arg_dtypes([None, None, None,
                             np.uint32, np.uint32, np.uint32])
def naive_func(a_g, b_g, c_g):
    P, Q = a_g.shape
    R = b_g.shape[1]
    start = time.time()
    naive_opencl(queue, (P, R), None, 
          a_g.data, b_g.data, c_g.data,
          P, Q, R)
    return time.time()-start

# Naive OpenCL algorithm using scalars for the middle values:
middle_opencl = cl.Program(ctx, """
#include <pyopencl-complex.h>
__kernel void func(__global const cfloat_t* a, __global const cfloat_t* b, 
                   __global cfloat_t* c,
                   const unsigned int P, const unsigned int Q, const unsigned int R) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);
    unsigned int k;
    cfloat_t tmp = cfloat_fromreal(0.0);

    for (k=0; k<Q; k++)
        tmp = cfloat_add(tmp, cfloat_mul(a[i*Q+k], b[k*R+j]));
    c[i*R+j] = tmp;
}
""").build().func
middle_opencl.set_scalar_arg_dtypes([None, None, None,
                             np.uint32, np.uint32, np.uint32])
def middle_opencl_func(a_g, b_g, c_g):
    P, Q = a_g.shape
    R = b_g.shape[1]
    start = time.time()
    middle_opencl(queue, (P, R), None, 
          a_g.data, b_g.data, c_g.data, P, Q, R)
    return time.time()-start

# Algorithm using local memory to store the columns of the second matrix.
local_mem = cl.Program(ctx, """
#include <pyopencl-complex.h>
__kernel void func(__global const cfloat_t* a, __global const cfloat_t* b, 
                   __global cfloat_t* c,
                   const unsigned int P, const unsigned int Q, const unsigned int R,
                   __local cfloat_t* b_col) {
    unsigned int i = get_global_id(0);
    unsigned int iloc = get_local_id(0);
    unsigned int nloc = get_local_size(0);
    unsigned int j, n;
    cfloat_t tmp;
    cfloat_t a_row[1024];

    for (n=0; n<Q; n++)
        a_row[n] = a[i*Q+n];
    for (j=0; j < R; j++) {
        for (n=iloc; n<Q; n+=nloc)
            b_col[n] = b[n*R+j];
        barrier(CLK_LOCAL_MEM_FENCE);
        tmp = cfloat_fromreal(0.0);
        for (n=0; n<Q; n++)
            tmp = cfloat_add(tmp, cfloat_mul(a_row[n], b_col[n]));
        c[i*R+j] = tmp;
    }
}
""").build().func
local_mem.set_scalar_arg_dtypes([None, None, None,
                             np.uint32, np.uint32, np.uint32, None])
def local_mem_func(a_g, b_g, c_g, local=32):
    P, Q = a_g.shape
    R = b_g.shape[1]
    b_col = cl.LocalMemory(np.complex64().nbytes*Q)
    start = time.time()
    local_mem(queue, (P,), (local,), 
          a_g.data, b_g.data, c_g.data, P, Q, R, b_col)
    return time.time()-start

# Create and initialize arrays for the four defined algorithms
def create_arrays(P, Q, R, S, T):
    a = (np.random.rand(P, Q)+1j*np.random.rand(P, Q)).astype(np.complex64)
    b = (np.random.rand(Q, R)+1j*np.random.rand(Q, R)).astype(np.complex64)
    c = np.empty((P, R), dtype=np.complex64)
    d = np.random.random((T, S)).astype(np.float32)
    d_trans = np.zeros((S,T)).astype(np.float32)
    return a, b, c, d, d_trans

def create_g(a, b, c):
    a_g = cl.array.to_device(queue, a)
    b_g = cl.array.to_device(queue, b)
    c_g = cl.array.empty(queue, (a_g.shape[0], b_g.shape[1]), a.dtype)
    return a_g, b_g, c_g

# Define the transpose function which uses memory buffers and queue to transpose matrix.
# This function outputs the time it takes for Python to run the algorithm vs. PyOpenCL
def transpose(A_trans, A, S, T, M):
    mf=cl.mem_flags                                                 
    a_buf=cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    atrans_buf=cl.Buffer(ctx, mf.WRITE_ONLY, A_trans.nbytes)
    cl.enqueue_copy(queue,atrans_buf,A_trans)  
    prg=cl.Program(ctx,kernel).build()       
    prg.mat_transpose(queue,A_trans.shape,None,a_buf,atrans_buf,np.uint32(T),np.uint32(S))
    cl.enqueue_copy(queue,A_trans,atrans_buf)      
    a_trans2 = np.zeros_like(A)
    a_trans2=np.transpose(A)
    times = []
    for i in xrange(M):
         start = time.time()
       	 y=np.transpose(d)
         times.append(time.time()-start)
         print 'Python time for Transpose Algorithm: ', np.average(times)

	 times = []
	 for i in xrange(M):
        	start = time.time()
        	prg.mat_transpose(queue,d_trans.shape,None,a_buf,atrans_buf,np.uint32(T),np.uint32(S))
        	times.append(time.time()-start)
	 print 'PyOpenCL time for Transpose Algorithm: ', np.average(times),'\n'

	 return a_trans2

# Initialize Matrix Dimensions
P = 32
Q = 64
R = 128
S = 5
T = 7
M = 3

a, b, c, d, d_trans = create_arrays(P, Q, R, S, T)
A_new = transpose(d_trans, d, S, T, 3)
a_g, b_g, c_g = create_g(a, b, c)

# Determine Python time and PyOpenCL Time
py_time = mult_py(a, b, c)
pyopencl_time_0 = naive_func(a_g, b_g, c_g)
c_0 = c_g.get()
pyopencl_time_1 = middle_opencl_func(a_g, b_g, c_g)
c_1 = c_g.get()
pyopencl_time_4 = local_mem_func(a_g, b_g, c_g, 2)
c_2 = c_g.get()

# Compare PyOpenCL (determined from GPU) to those with Python
print 'Are Python and PyOpenCL Algorithms equal?'
print 'Naive OpenCL = Python: ', np.allclose(c, c_0)
print 'Naive OpenCL with Scalars as Middle Values = Python: ', np.allclose(c, c_1)
print 'OpenCL using Local Memory = Python: ', np.allclose(c, c_2)
print 'Transpose: ', np.allclose(d_trans, A_new)

def python_time(P, Q, R, M=3):
    times = []
    a, b, c, d, d_trans = create_arrays(P, Q, R, S, T)
    for i in xrange(M):
        t = mult_py(a, b, c)
        times.append(t)
    return np.average(times)

def pyopencl_time(f, P, Q, R, M=3):
    times = []
    a, b, c, d, d_trans = create_arrays(P, Q, R, S, T)
    a_g, b_g, c_g = create_g(a, b, c)
    for i in xrange(M):
        t = f(a_g, b_g, c_g)
        times.append(t)
    return np.average(times)
time_0 = lambda P, Q, R, M=3: pyopencl_time(naive_func, P, Q, R, M)
time_1 = lambda P, Q, R, M=3: pyopencl_time(middle_opencl_func, P, Q, R, M)
time_2 = lambda P, Q, R, M=3: pyopencl_time(local_mem_func, P, Q, R, M)

# Determine the speed of PyOpenCL and Python vs the Number of Elements in the Array
# Plot results
time_0_list = []
time_1_list = []
time_2_list = []
i_range = range(1, 11, 1)
for i in i_range:
    time_0_list.append(time_0(i*P, i*Q, i*R, M))
    time_1_list.append(time_1(i*P, i*Q, i*R, M))
    time_2_list.append(time_2(i*P, i*Q, i*R, M))

plt.clf()
a_size_list = np.array(i_range)*P*Q
b_size_list = np.array(i_range)*Q*R
plt.subplot(211)
plt.plot(a_size_list, time_0_list, 'bo-',
         a_size_list, time_1_list, 'ro-',
         a_size_list, time_2_list, 'ko-')
plt.xlabel('# Elements in Array: 32 by 64')
plt.ylabel('$t$')
plt.title('Different Implementations')
plt.legend(('Naive', 'Scalars as Middle Values', 'Local Memory'), loc='upper left')
plt.grid(True)
plt.gca().set_xlim((min(a_size_list), max(a_size_list)))
plt.subplot(212)
plt.plot(b_size_list, time_0_list, 'bo-',
         b_size_list, time_1_list, 'ro-',
         b_size_list, time_2_list, 'ko-')
plt.xlabel('# Elements in Array: 64 by 128')
plt.ylabel('$t$')
plt.grid(True)
plt.gca().set_xlim((min(b_size_list), max(b_size_list)))
plt.draw()
plt.savefig('scaling.png')

with open('scaling.csv', 'w') as f:
    w = csv.writer(f)
    for a_size, b_size, t_0, t_1, t_2 in \
        zip(a_size_list, b_size_list,
            time_0_list, time_1_list,
            time_2_list):
        w.writerow([a_size, b_size, t_0, t_1, t_2])
