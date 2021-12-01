// Main

#include "cl_helper.hpp"

int main(int argc, char** argv) {
    
    // Setting up memory for contextx and platforms
    cl_uint num_platforms, num_devices;
    
    cl_device_id *devices;
    cl_context *contexts;
    cl_platform_id *platforms;
    cl_int ret_code = CL_SUCCESS;
    
    // Could be CL_DEVICE_TYPE_GPU or CL_DEVICE_TYPE_CPU
    cl_device_type device_type = CL_DEVICE_TYPE_ALL;
    
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
            CL_TRUE);

    // Release command queues
    h_release_command_queues(command_queues, num_command_queues);

    // Release devices and contexts
    h_release_devices(devices, num_devices, contexts, platforms);
}
