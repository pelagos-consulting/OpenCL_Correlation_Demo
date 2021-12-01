// Main

#include <assert.h>
#include "cl_helper.hpp"
#include <cstdio>
#include <omp.h>
#include <chrono>

#define N0 1024
#define N1 1280
#define NIMAGES 1024
#define L0 0 
#define R0 2
#define L1 0
#define R1 2

int main(int argc, char** argv) {
    
    // Setting up memory for contextx and platforms
    cl_uint num_platforms, num_devices;
    
    cl_device_id *devices;
    cl_context *contexts;
    cl_platform_id *platforms;
    cl_int ret_code = CL_SUCCESS;
    
    // Could be CL_DEVICE_TYPE_GPU or CL_DEVICE_TYPE_CPU
    cl_device_type device_type = CL_DEVICE_TYPE_ALL;
    
    // Get the device to use
    if (argc>1) {
        if (strcmp(argv[1], "GPU")==0) {
            // We get the GPU type
            device_type = CL_DEVICE_TYPE_GPU;
        } else if (strcmp(argv[1], "CPU")==0) {
            // We get the CPU type
            device_type = CL_DEVICE_TYPE_CPU;
        } 
        // Else we get the ALL type
    }
    
    // Get iteration count
    cl_int NITERS = 10;
    if (argc>2) {
        NITERS = atoi(argv[2]);
    }
    
    
    // Get devices and contexts
    h_acquire_devices(device_type, 
                    &platforms, &num_platforms, 
                    &devices, &num_devices,
                    &contexts);
    
    // Make a command queue and report on devices
    for (cl_uint n=0; n<num_devices; n++) {
        h_report_on_device(devices[n]);
    }
   
    // Create command queues 
    cl_uint num_command_queues = num_devices;
    cl_command_queue* command_queues = h_create_command_queues(
            devices,
            contexts,
            num_devices,
            num_command_queues,
            CL_FALSE,
            CL_FALSE);

    // Create memory for images out
    float* images_out = (float*)calloc(NIMAGES*N0*N1, sizeof(float));
    
    // images_in will have dimensions (NIMAGES, N0, N1) and will have row-major ordering

    // Read in images
    size_t nbytes;
    float* images_in = (float*)h_read_file("images_in.dat", "rb", &nbytes);
    assert(nbytes == NIMAGES*N0*N1*sizeof(float));

    // Read in image Kernel
    size_t nelements_image_kernel = (L0+R0+1)*(L1+R1+1);
    float* image_kernel = (float*)h_read_file("image_kernel.dat", "rb", &nbytes);
    assert(nbytes == nelements_image_kernel*sizeof(float));

    // Read kernel sources 
    const char* filename = "kernels.cl";
    char* source = (char*)h_read_file(filename, "r", &nbytes);

    // Create Programs and kernels using this source
    cl_program *programs = (cl_program*)calloc(num_devices, sizeof(cl_program));
    cl_kernel *kernels = (cl_kernel*)calloc(num_devices, sizeof(cl_kernel));
    
    for (cl_uint n=0; n<num_devices; n++) {
        // Make the program from source
        programs[n] = h_build_program(source, contexts[n], devices[n]);
        // And make the kernel
        kernels[n] = clCreateKernel(programs[n], "xcorr", &ret_code);
        h_errchk(ret_code, "Making a kernel");
    }

    // Create memory for images in and images out for each device
    cl_mem *buffer_srces = (cl_mem*)calloc(num_devices, sizeof(cl_mem));
    cl_mem *buffer_dests = (cl_mem*)calloc(num_devices, sizeof(cl_mem));
    cl_mem *buffer_kerns = (cl_mem*)calloc(num_devices, sizeof(cl_mem));
   
    // Create buffers
    for (cl_uint n=0; n<num_devices; n++) {
        // Create buffers for sources
        buffer_srces[n] = clCreateBuffer(
                contexts[n],
                CL_MEM_READ_WRITE,
                N0*N1*sizeof(float),
                NULL,
                &ret_code);
        h_errchk(ret_code, "Creating buffers for sources");

        // Create buffers for destination
        buffer_dests[n] = clCreateBuffer(
                contexts[n],
                CL_MEM_READ_WRITE,
                N0*N1*sizeof(float),
                NULL,
                &ret_code);
        h_errchk(ret_code, "Creating buffers for destinations");

        // Copy host memory for the image kernel
        buffer_kerns[n] = clCreateBuffer(
                contexts[n],
                CL_MEM_COPY_HOST_PTR,
                nelements_image_kernel*sizeof(float),
                (void*)image_kernel,
                &ret_code);
        h_errchk(ret_code, "Creating buffers for image kernel");

        // Just for kernel arguments
        cl_int len0_src = N0, len1_src = N1, pad0_l = L0, pad0_r = R0, pad1_l = L1, pad1_r = R1;

        // Set kernel arguments here for convenience
        h_errchk(clSetKernelArg(kernels[n], 0, sizeof(buffer_srces[n]), &buffer_srces[n]), "Set kernel argument 0");
        h_errchk(clSetKernelArg(kernels[n], 1, sizeof(buffer_dests[n]), &buffer_dests[n]), "Set kernel argument 1");
        h_errchk(clSetKernelArg(kernels[n], 2, sizeof(buffer_kerns[n]), &buffer_kerns[n]), "Set kernel argument 2");
        h_errchk(clSetKernelArg(kernels[n], 3, sizeof(cl_int), &len0_src),  "Set kernel argument 3");
        h_errchk(clSetKernelArg(kernels[n], 4, sizeof(cl_int), &len1_src),  "Set kernel argument 4");
        h_errchk(clSetKernelArg(kernels[n], 5, sizeof(cl_int), &pad0_l),    "Set kernel argument 5");
        h_errchk(clSetKernelArg(kernels[n], 6, sizeof(cl_int), &pad0_r),    "Set kernel argument 6");
        h_errchk(clSetKernelArg(kernels[n], 7, sizeof(cl_int), &pad1_l),    "Set kernel argument 7");
        h_errchk(clSetKernelArg(kernels[n], 8, sizeof(cl_int), &pad1_r),    "Set kernel argument 8");
    }

    // Use OpenMP to dynamically distribute threads across the available workflow of images
    //omp_set_dynamic(0);
    //omp_set_num_threads(num_devices);
    
    // This counter keeps track of images process by all iterations
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    
    cl_uint* it_count = (cl_uint*)calloc(num_devices, sizeof(cl_uint)); 

    for (cl_uint i = 0; i<NITERS; i++) {
        printf("Processing iteration %d of %d\n", i+1, NITERS);
        
        #pragma omp parallel for default(none) schedule(dynamic, 1) num_threads(num_devices) \
            shared(images_in, buffer_dests, buffer_srces, \
                    images_out, image_kernel, nelements_image_kernel, \
                    command_queues, kernels, buffer_kerns, it_count)
        for (cl_uint n=0; n<NIMAGES; n++) {
            // Get the thread_id
            int tid = omp_get_thread_num();
            it_count[tid] += 1;
            
            // Load memory from images in using the offset
            size_t offset = n*N0*N1;

            //printf("Processing image %d of %d with device %d\n", n+1, NIMAGES, tid);
            
            // Write from main memory to the buffer
            h_errchk(clEnqueueWriteBuffer(
                        command_queues[tid],
                        buffer_srces[tid],
                        CL_FALSE,
                        0,
                        N0*N1*sizeof(float),
                        images_in + offset,
                        0,
                        NULL,
                        NULL), "Writing to source buffer");
            
            // Upload the images kernel
            h_errchk(clEnqueueWriteBuffer(
                        command_queues[tid],
                        buffer_kerns[tid],
                        CL_FALSE,
                        0,
                        nelements_image_kernel*sizeof(float),
                        image_kernel,
                        0,
                        NULL,
                        NULL), "Writing to image kernel buffer");

            // Enqueue the kernel
            cl_uint work_dims = 2;
            const size_t local_size[] = {16, 16};

            size_t gs_0 = N0/local_size[0];
            if (N0 % local_size[0] > 0) gs_0 += 1;
            size_t gs_1 = N1/local_size[1];
            if (N1 % local_size[1] > 0) gs_1 += 1;
            const size_t global_size[] = {gs_0*local_size[0], gs_1*local_size[1]};

            h_errchk(clEnqueueNDRangeKernel(
                        command_queues[tid],
                        kernels[tid],
                        work_dims,
                        NULL,
                        global_size,
                        local_size,
                        0, 
                        NULL,
                        NULL), "Running the xcorr kernel");

            // Read from the buffer to main memory and block
            h_errchk(clEnqueueReadBuffer(
                        command_queues[tid],
                        buffer_dests[tid],
                        CL_TRUE,
                        0,
                        N0*N1*sizeof(float),
                        images_out + offset,
                        0,
                        NULL,
                        NULL), "Writing to buffer");
        }
    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
    double duration = time_span.count();
    
    cl_uint num_images = NITERS*NIMAGES;
    for (cl_uint i = 0; i< num_devices; i++) {
        //h_report_on_device(devices[i]);
        float pct = 100*float(it_count[i])/float(num_images);
        printf("Device %d processed %d of %d images (%0.2f\%)\n", i, it_count[i], num_images, pct);
    }
    printf("Overall processing rate %0.2f images/s\n", (double)num_images/duration);

    // Write output data to output file
    FILE* fp = fopen("images_out.dat", "w+b");
    fwrite(images_out, sizeof(float), NIMAGES*N0*N1, fp);
    fclose(fp);

    // Free memory
    free(source);
    free(image_kernel);
    free(images_in);
    free(images_out);
    free(it_count);

    // Release command queues
    h_release_command_queues(command_queues, num_command_queues);

    // Release programs and kernels
    for (cl_uint n=0; n<num_devices; n++) {
        h_errchk(clReleaseKernel(kernels[n]), "Releasing kernel");
        h_errchk(clReleaseProgram(programs[n]), "Releasing program");
        h_errchk(clReleaseMemObject(buffer_srces[n]),"Releasing sources buffer");
        h_errchk(clReleaseMemObject(buffer_dests[n]),"Releasing dests buffer");
        h_errchk(clReleaseMemObject(buffer_kerns[n]),"Releasing image kernels buffer");
    }

    // Free memory
    free(buffer_srces);
    free(buffer_dests);
    free(buffer_kerns);
    free(programs);
    free(kernels);

    // Release devices and contexts
    h_release_devices(devices, num_devices, contexts, platforms);
}
