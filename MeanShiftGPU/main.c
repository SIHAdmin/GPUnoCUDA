// Demonstration of GPU accelerated MeanShift clustering
// - Depends on GNU statistics library (https://www.gnu.org/software/gsl/)
//
// Adapted from https://spin.atomicobject.com/2015/06/02/opencl-c-mac-osx/ (Copyright 2015 Atomic Object MIT license)
/*
The MIT License (MIT)

Copyright 2015 Atomic Object

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
//
// CC BY-SA, Hayim Dar 2018

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef __MACH__
#include <sys/time.h>
#include <mach/clock.h>
#include <mach/mach.h>
#endif

#include <OpenCL/opencl.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

#include "mean_shift_point.cl.h"        // ** THIS IS WHERE WE INCLUDE THE OpwnCL GPU KERNEL FUNCTION! **

//#define VERIFY_OPENCL_OUTPUT
//#define USE_CPU


// SUBFUNCTIONS
// ------------
// a. Verification function --> UNSUSED
static int verify_mean_shift(cl_float2 *points, cl_float2 *original_points,
                             size_t num_points, cl_float bandwidth,
                             cl_float2 *shifted_points);

// b. OpenCL Device query and printout function
static void print_device_info(cl_device_id device) {
    char name[128];
    char vendor[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, 128, name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, 128, vendor, NULL);
    fprintf(stdout, "%s : %s\n", vendor, name);
}

// c. Minimum and maximum functions.. because C is a terrible langauge to write maths in!
float min_array( float arr[], int len )
{
    float min = arr[0];
    for ( int i = 1; i < len; i++ )
    if ( arr[i] < min )
    min = arr[i];
    return min;
}
float max_array( float arr[], int len )
{
    float max = arr[0];
    for ( int i = 1; i < len; i++ )
    if ( arr[i] > max )
    max = arr[i];
    return max;
}

// d. Unique elements function.. because C is a terrible langauge to write maths in!
float *unique_elements(float arr[], int len, int *ulen) {

    float counted[len];
    int j, n, count, flag;

    counted[0] = arr[0];
    count = 1;/*one element is counted*/

    for(j=0; j <= len-1; ++j) {
        flag = 1;;
        /*the counted array will always have 'count' elements*/
        for(n=0; n < count; ++n) {
            if( fabs(arr[j] - counted[n])/arr[j] < .01) {
                flag = 0;
            }
        }
        if(flag == 1) {
            ++count;
            counted[count-1] = arr[j];
        }
    }
    *ulen = count;
    float *uarr = malloc(count*sizeof(float));
    memcpy(uarr,counted,count*sizeof(float));
    return uarr;
}

// e. Deal with the face that OSX versions may or may not have clock_gettime
void current_utc_time(struct timespec *ts) {
#ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    clock_get_time(cclock, &mts);
    mach_port_deallocate(mach_task_self(), cclock);
    ts->tv_sec = mts.tv_sec;
    ts->tv_nsec = mts.tv_nsec;
#else
    clock_gettime(CLOCK_REALTIME, ts);
#endif
}
// ------------


// Main program function!
// =====================
int main (int argc, const char *argv[]) {
    // --> Deal with arguments
    if( argc > 2 ) {
        printf("ERROR> Too many arguments supplied.\n");
        return 1;
    }

    // --> This is to measure elapsed wall time
    struct timespec now, tmstart;

    // Check if GNUplot is installed
    int DO_PLOTS;
    if (system("gnuplot --version gnuplot > /dev/null 2>&1") == 0) {
        fprintf(stdout, "GNU plot found, will use for figures..\n\n");
        DO_PLOTS = 1;
    }
    else {
        fprintf(stdout, "WARNING> GNUplot not found; will not plot results.\n\n");
        DO_PLOTS = 0;
    }


// Initialise OpenCL environment
// -----------------------------
    // --> Query openCL devices
    // OpenCl uses 'contexts' to define the devices/environments it will compute on
    cl_context context = gcl_get_context();
    size_t length;
    cl_device_id devices[8];
    clGetContextInfo(
                     context, CL_CONTEXT_DEVICES, sizeof(devices), devices, &length);
    fprintf(stdout, "The following devices are available for use:\n");

    // print device info
    int num_devices = (int)(length / sizeof(cl_device_id));
    int i;
    for (i = 0; i < num_devices; i++) {
        fprintf(stdout, "%d) ", i);
        print_device_info(devices[i]);
    }

    // --> Which OpenCL device to use?
    int devID = -1;
    while ( devID < 0 || devID > num_devices ) {
        fprintf(stdout, "Please select a device to run on: ");
        scanf("%d",&devID);
        //fprintf(stdout, "Selected device %d.\n", devID);
    }

    // --> Start timer, for benchmarking
    current_utc_time(&tmstart);

    // --> Create dispatch queue
    // Calculations are sent to devices via 'queues'
    dispatch_queue_t queue =
#ifdef USE_CPU
    // the USE_CPU flag circumvents the OpenCl framework entirely, to run directly on the CPU. We will not use this!
    NULL;
#else
    // Instead, we will create a queue on the device selected above, which may also be the CPU
    gcl_create_dispatch_queue(CL_DEVICE_TYPE_USE_ID, devices[devID]);
#endif

    if (queue == NULL) {
        queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_CPU, NULL);;
        fprintf(stderr, "Warning: Running on CPU\n");
    }
    else {
        char name[128];
        cl_device_id gpu = gcl_get_device_id_with_dispatch_queue(queue);
        clGetDeviceInfo(gpu, CL_DEVICE_NAME, 128, name, NULL);
        fprintf(stderr, "> Running on %s\n", name);
    }
// -----------------------------



// Generate input data
// -------------------
    // Set the size of the sample
    const int NUM_VALUES = 128 * 160 / 4;
    const int BUFFER_SIZE = sizeof(cl_float2) * NUM_VALUES;

    // also set the maxmimum number of mean-shift iterations
    const int MAX_ITERATIONS = 100;

    // --> Allocate Host memory (normal addressable array allocations)
    // cl_float2 is an OpenCL defined datatype which is a tuple (x,y); makes the code a little neater!
    cl_float2 *points = (cl_float2 *)malloc(BUFFER_SIZE);
    cl_float2 *original_points = (cl_float2 *)malloc(BUFFER_SIZE);
    cl_float2 *shifted_points = (cl_float2 *)malloc(BUFFER_SIZE);
    float *modes = (float *)malloc(BUFFER_SIZE/2);


    // Data will be sampled from a number of gaussian distributions, generating the clusters:
    // --> Set random seed
    time_t t_seed;
    if ( argv[1] == NULL) {
        t_seed = time(NULL);
    }
    else {
        t_seed = atoi(argv[1]);
    }
    printf("\nRandom seed %ld\n", t_seed);
    srand((unsigned int) t_seed); // demo: 1515482054,1513833180, 1513834108, 1515629158

    // --> Choose number of clusters between [2,5]
    int Nc = 2 + rand() % 4;

    // --> Generate distribution sigmas between [1,3], and centres, independently for the x and y dimensions
    float sig[2][Nc];
    float mu[2][Nc];
    for (int i = 0; i < Nc; i++) {
        sig[0][i] = (float) (1 + rand() % 10) / 10; // sigma from [0.2, 2]
        sig[1][i] = (float) (1 + rand() % 10) / 10; // sigma from [0.2, 2]
        mu[0][i] = (rand() % 100) / 10;
        mu[1][i] = (rand() % 100) / 10;
    }

    // --> Use Gaussian sampler from the GNU scientific library (gsl) to sample for the clusters
    // (which uses the gsl random number generator)
    gsl_rng *r;
    r = gsl_rng_alloc(gsl_rng_mt19937);
    int j;
    for (int i = 0; i < NUM_VALUES; i++) {
        // Simply use a modulo function (%) to assign data points to different clusters equally
        j = i % Nc;
        points[i].x = (cl_float) mu[0][j] + gsl_ran_gaussian(r, sig[0][j]);
        points[i].y = (cl_float) mu[1][j] + gsl_ran_gaussian(r, sig[1][j]);
    }

    // --> Set the convolution bandwidth
    // reshape the sigmas vector so I can find the min and max of them..
    float (*resig)[1] = (float (*)[1])sig;
    // cl_float BANDWIDTH = 0.20* (max_array( *resig , 2*Nc) + min_array( *resig , 2*Nc));
    cl_float BANDWIDTH = 2.5* min_array( *resig , 2*Nc);
    fprintf(stdout, "Bandwidth is %f.\n",BANDWIDTH);

    // free the gsl random object
    gsl_rng_free(r);

    // --> Write input data to file
    FILE *temp = fopen("data.temp", "w");
    for (int i=0; i<NUM_VALUES; i++) {
        fprintf(temp, "%lf %lf %d\n", points[i].x, points[i].y, i%Nc + 1);
    }
    fclose(temp);


    if (DO_PLOTS == 1) {
    // Plot data via GNUplot
    FILE *gnuplotPipe = popen("gnuplot -persistent", "w");
    fprintf(gnuplotPipe, "set title \"%d generating clusters (seed = %ld)\"\n",Nc,t_seed);
    fprintf(gnuplotPipe, "set xrange [-1:1]; set yrange [-1:1]; set cbrange [-1:1]\n");
    fprintf(gnuplotPipe, "plot \"<echo '0 0 0'\" with points pt 0 lw 0 palette notitle\n");
    fprintf(gnuplotPipe, "set autoscale xy; set autoscale cb\n");
    fprintf(gnuplotPipe, "replot 'data.temp' with points pt 6 lw .5 palette\n");
    fflush(gnuplotPipe);
    }
// ----------------


// Mean-shift algorithm in OpenCL
// ------------------------------
    // --> First make a copy of the dample data
    memcpy(original_points, points, BUFFER_SIZE);

    // --> Allocate OpenCL device memory buffers (these are not directly addressable by code on the host)
    void *device_points = gcl_malloc(BUFFER_SIZE, points,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void *device_original_points = gcl_malloc(BUFFER_SIZE, original_points,
                                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void *device_shifted_points = gcl_malloc(BUFFER_SIZE, NULL, CL_MEM_WRITE_ONLY);

    // --> Schedule work on the OpenCL dispatch queue
    __block int num_work_groups = 0;
    int iteration = 0;

    // Loop over iterations..
    for (;;) {
        // Dispatch is normally async, but this demo doesn't have other
        // work to do while waiting for the GPU. Also be careful trying to
        // do other work if OpenCL is using the CPU--it doesn't leave any
        // unused cores by default.

        dispatch_sync(queue, ^{
            size_t work_group_size;                                         // Work group size is the number of items each compute unit ('local workgroup') will process
            gcl_get_kernel_block_workgroup_info(mean_shift_point_kernel,    // This function asks OpenCL to estimate what this work_group_size should be, for us;
                                                CL_KERNEL_WORK_GROUP_SIZE,  // we'll then pass this to the cl_ndrange object below.
                                                sizeof(work_group_size),
                                                &work_group_size, NULL);

            // --> This is the N-dimensional range over which to execute the Kernel (can be [1:3])
            cl_ndrange range = {
                1,                              // Since our data buffers are 1D, we weill use a 1D range
                { 0, 0, 0 },                    // This specifies an offset in each dimension; always specifiy all dims, even if not using
                { NUM_VALUES, 0, 0 },           // "global size": the total number of items to process in each dim
                { work_group_size, 0, 0 }       // "local size" of workgroups: which is the number of items each workgroup will process. This determines
            };                                  // This determines the number of workgroups also, since workgroups = global size / local size, as below:

            num_work_groups = NUM_VALUES / work_group_size;

            mean_shift_point_kernel(&range, (cl_float2 *)device_points,     // This is the actual computation, as defined by our Kernel function!
                                    (cl_float2 *)device_original_points,    // The function is defined in < mean_shift_point.cl >.
                                    NUM_VALUES, BANDWIDTH,
                                    (cl_float2 *)device_shifted_points);

            // --> Copy the computed points from the OpenCl device back to the host memory
            gcl_memcpy(shifted_points, device_shifted_points, BUFFER_SIZE);
        });
                                                                                        // Question: Why cant we copy the buffers straight over on the device?
        // --> If below max iterations,                                                 // Why the need to bring them back?
        if (++iteration < MAX_ITERATIONS) {
            // use the shifted points from above as the new input points (ie, shift them!)
            memcpy(points, shifted_points, BUFFER_SIZE);

            // Free the previous points' memory on the device, and copy over these new input points
            gcl_free(device_points);
            device_points = gcl_malloc(BUFFER_SIZE, points,
                                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
        }
        // --> else, we're done.
        else {
            break;
        }
    }

// --> Summary info
    // Computations done
    fprintf(stderr, "\n%d Iterations on %d work groups: Mean shifted %d points\n",
            iteration, num_work_groups, NUM_VALUES);

    // Walltime elapsed
    current_utc_time(&now);
    double seconds = (double)((now.tv_sec+now.tv_nsec*1e-9) - (double)(tmstart.tv_sec+tmstart.tv_nsec*1e-9));
    printf("Wall time %fs\n", seconds);

// --> Free device memory
    gcl_free(device_points);
    gcl_free(device_original_points);
    gcl_free(device_shifted_points);

// --> Release dispatch queue
    dispatch_release(queue);

// --> Data verification routine -- UNUSED
    int status = 0;
#ifdef VERIFY_OPENCL_OUTPUT
    if (!verify_mean_shift(points, original_points, NUM_VALUES, BANDWIDTH,
                           shifted_points))
    {
        fprintf(stderr, "Values were not computed properly!\n");
        status = 1;
    }
#endif
// --------------------------------


// Compute clusters
// ----------------
    // -> Find convergence points (by simply mulitplying the x,y values.. not robust!)
    temp = fopen("result.temp", "w");
    for (int i=0; i<NUM_VALUES; i++) {
        modes[i] = log( 1 + fabs(shifted_points[i].x * shifted_points[i].y));
    }
    // -> Select unique values (> nmodes)
    float *modexy; int nmodes;
    modexy = unique_elements(modes, NUM_VALUES, &nmodes);

    // --> Count cluster sizes (> pops)
    // label each data point via its membership (ie convergence) to the unique stationary points above (> cluster)
    int *cluster = (int *)malloc(BUFFER_SIZE/2);
    int pops[nmodes]; memset( pops, 0, sizeof(pops));
    for (int k=0; k<NUM_VALUES; k++) {
        for  (int i=0; i<nmodes; i++) {
            if ( fabs(modes[k] - modexy[i])/modes[k] < 0.01 ) {
                cluster[k] = i;
                pops[i]++;
                break;
            }
        }
    }
    free(modexy); // since it was allocated 'in stack'

    // --> Find clusters of at least minimum size = .75 of actual population
    // and assign them an index (> big_index), which will also be used for false-colouring the GNU-plots
    fprintf(stdout, "\n");
    float *clustdx = (float *)malloc(sizeof(float)*nmodes);
    int big_index = 0;
    for  (int i=0; i<nmodes; i++) {
        if (pops[i] >= NUM_VALUES/nmodes*.75) {
            big_index++;
            clustdx[i] = big_index;
        }
        // else assign a fractional index (this is to make the gnuplot colouring easier)
        else {
            clustdx[i] = (float) i / nmodes / 10;
        }
        // Print out the cluster IDs and populations
        fprintf(stdout, "clustdx[%d] = %f. (pop=%d)\n", i, clustdx[i], pops[i]);
    }

    // --> Re-label according to accepted clusters above (> cluster2)
    float *cluster2 = (float *)malloc(sizeof(float)*NUM_VALUES);
    for (int k=0; k<NUM_VALUES; k++) {
        cluster2[k] = clustdx[cluster[k]];
        // Write results to result file
        fprintf(temp, "%lf %lf %lf %lf %d %d %f\n", original_points[k].x, original_points[k].y, shifted_points[k].x, shifted_points[k].y, k%Nc + 1, cluster[k] + 1, cluster2[k]);
    }
    fclose(temp);

    if (DO_PLOTS == 1) {
    // Plot outputs via GNUplot
    FILE *gnuplotPipe = popen("gnuplot -persistent", "w");
    gnuplotPipe = popen("gnuplot -persistent", "w");
    fprintf(gnuplotPipe, "set title \"Results: %d clusters (bandwidth = %2f)\"\n",nmodes,BANDWIDTH);
    fprintf(gnuplotPipe, "set xrange [-1:1]; set yrange [-1:1]; set cbrange [-1:1]\n");
    fprintf(gnuplotPipe, "plot \"<echo '0 0 0'\" with points pt 0 lw 0 palette notitle\n");
    fprintf(gnuplotPipe, "set autoscale xy; set autoscale cb\n");
    fprintf(gnuplotPipe, "replot 'result.temp' using 1:2:7 with points lw 1 palette\n");
    fflush(gnuplotPipe);
    fclose(gnuplotPipe);
    }
// ----------------


// Release host mem
    free(points);
    free(original_points);
    free(shifted_points);

    fprintf(stdout, "\nDone.\n");
    exit(status);
}
// END.


// More SUBFUNCTION definitions (> verify-mean_shift)
// ----------------------------
static cl_float euclidian_distance(cl_float2 p1, cl_float2 p2) {
    return sqrtf(powf(p1.x - p2.x, 2.f) +
                 powf(p1.y - p2.y, 2.f));
}

static cl_float gaussian_kernel(cl_float dist, cl_float bandwidth) {
    return (1.f / (bandwidth * sqrtf(2.f * (float)M_PI))) *
            expf(-0.5f * powf(dist / bandwidth, 2.f));
}

static int verify_mean_shift(cl_float2 *points, cl_float2 *original_points,
                             size_t num_points, cl_float bandwidth,
                             cl_float2 *shifted_points)
{
    for (int i = 0; i < num_points; i++) {
        cl_float2 shift = { 0, 0 };
        cl_float scale = 0;

        for (int j = 0; j < num_points; j++) {
            cl_float dist = euclidian_distance(points[i], original_points[j]);
            cl_float weight = gaussian_kernel(dist, bandwidth);

            shift.x += original_points[j].x * weight;
            shift.y += original_points[j].y * weight;
            scale += weight;
        }

        cl_float2 expected = { shift.x / scale, shift.y / scale };

        if (fabs(shifted_points[i].x - expected.x) > 0.01 ||
            fabs(shifted_points[i].y - expected.y) > 0.01)
        {
            fprintf(stdout, "Error: Element %d did not match expected output.\n", i);
            fprintf(stdout, "       Saw (%1.8f,%1.8f), expected (%1.8f,%1.8f)\n",
                    shifted_points[i].x, shifted_points[i].y, expected.x, expected.y);
            fflush(stdout);
            return 0;
        }
    }

    return 1;
}
