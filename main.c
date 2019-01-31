//  DYNAMIC MEAN FIELD MODEL Deco et al. 2014 Journal of Neuroscience
//
//  Code created by Michael Schirner on 12.08.16
//  m.schirner@fu-berlin.de or michael.schirner@charite.de
//  Copyright (c) 2016 Michael Schirner. All rights reserved.


#include <stdio.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
//#include "mpi.h"

struct Xi_p{
    float **Xi_elems;
};

struct SC_capS{
    float *cap;
};

struct SC_inpregS{
    int *inpreg;
};

FILE *FCout, *WFout;

#define REAL float
//#define REAL double


/* Compute Pearson's correlation coefficient */
float corr(float *x, float *y, int n){
    int i;
    float mx=0, my=0;
    
    /* Calculate the mean of the two series x[], y[] */
    for (i=0; i<n; i++) {
        mx += x[i];
        my += y[i];
    }
    mx /= n;
    my /= n;
    
    /* Calculate the correlation */
    float sxy = 0, sxsq = 0, sysq = 0, tmpx, tmpy;
    for (i=0; i<n; i++) {
        tmpx = x[i] - mx;
        tmpy = y[i] - my;
        sxy += tmpx*tmpy;
        sxsq += tmpx*tmpx;
        sysq += tmpy*tmpy;
    }
    
    return (sxy / (sqrt(sxsq)*sqrt(sysq)));
}



void openFCoutfile(char *paramset){
    char outfilename[1000];memset(outfilename, 0, 1000*sizeof(char));
    char buffer[10];memset(buffer, 0, 10*sizeof(char));
    char underscore[2];
    underscore[0]='_';
    underscore[1]='\0';
    strcpy (outfilename,"output/");
    strcat (outfilename,"/BOLD_");strcat (outfilename,paramset);
    strcat (outfilename,".txt");
    FCout = fopen(outfilename, "w");
    memset(outfilename, 0, 1000*sizeof(char));
    strcpy (outfilename,"output/");
    strcat (outfilename,"/SV_");strcat (outfilename,paramset);
    strcat (outfilename,".txt");
    //WFout = fopen(outfilename, "w");
}



float gaussrand_ret()
{
    static double V1=0.0, V2=0.0, S=0.0, U1=0.0, U2=0.0;
    S=0.0;
    do {
        U1 = (double)rand() / RAND_MAX;
        U2 = (double)rand() / RAND_MAX;
        V1 = 2 * U1 - 1;
        V2 = 2 * U2 - 1;
        S = V1 * V1 + V2 * V2;
    } while(S >= 1 || S == 0);
    
    return (float)(V1 * sqrt(-2 * log(S) / S));
}

static inline void gaussrand(float *randnum)
{
    static double V1=0.0, V2=0.0, S=0.0, U1=0.0, U2=0.0;
    
    S=0.0;
    do {
        U1 = (double)rand() / RAND_MAX;
        U2 = (double)rand() / RAND_MAX;
        V1 = 2 * U1 - 1;
        V2 = 2 * U2 - 1;
        S = V1 * V1 + V2 * V2;
    } while(S >= 1 || S == 0);
    
    randnum[0] = (float)(V1 * sqrt(-2 * log(S) / S));
    randnum[1] = (float)(V2 * sqrt(-2 * log(S) / S));
    
    S=0.0;
    do {
        U1 = (double)rand() / RAND_MAX;
        U2 = (double)rand() / RAND_MAX;
        V1 = 2 * U1 - 1;
        V2 = 2 * U2 - 1;
        S = V1 * V1 + V2 * V2;
    } while(S >= 1 || S == 0);
    
    randnum[2] = (float)(V1 * sqrt(-2 * log(S) / S));
    randnum[3] = (float)(V2 * sqrt(-2 * log(S) / S));
}


int importGlobalConnectivity(char *SC_cap_filename, char *SC_dist_filename, char *SC_inputreg_filename, int regions, float **region_activity, struct Xi_p **reg_globinp_p, float global_trans_v, int **n_conn_table, float **n_conn_table_G_NMDA, float G_J_NMDA, struct SC_capS **SC_cap, float **SC_rowsums, struct SC_inpregS **SC_inpreg)
{
    
    int i,j,k, tmp3, maxdelay=0, tmpint;
    float *region_activity_p;
    double tmp, tmp2;
    struct Xi_p *reg_globinp_pp;
    struct SC_capS      *SC_capp;
    struct SC_inpregS   *SC_inpregp;
    int num_incoming_conn_cap=0, num_incoming_conn_dist=0, num_incoming_conn_inputreg=0;
    
    // Open SC files
    FILE *file_cap, *file_dist, *file_inputreg;
    file_cap=fopen(SC_cap_filename, "r");
    file_dist=fopen(SC_dist_filename, "r");
    file_inputreg=fopen(SC_inputreg_filename, "r");
    if (file_cap==NULL || file_dist==NULL || file_inputreg==NULL)
    {
        printf( "\nERROR: Could not open SC files. Terminating... \n\n");
        exit(0);
    }
    
    // Read number of regions in header and check whether it's consistent with other specifications
    int readSC_cap, readSC_dist, readSC_inp;
    if(fscanf(file_cap,"%d",&readSC_cap) == EOF || fscanf(file_dist,"%d",&readSC_dist) == EOF || fscanf(file_inputreg,"%d",&readSC_inp) == EOF){
        printf( "\nERROR: Unexpected end-of-file in file. File contains less input than expected. Terminating... \n\n");
        exit(0);
    }
    if (readSC_cap != regions || readSC_dist != regions || readSC_inp != regions) {
        printf( "\nERROR: Inconsistent number of large-scale regions in SC files. Terminating... \n\n");
        fclose(file_dist);fclose(file_inputreg);
        exit(0);
    }
    
    // Allocate a counter that stores number of region input connections for each region and the SCcap array
    *SC_rowsums = (float *)_mm_malloc(regions*sizeof(float),16);
    *n_conn_table = (int *)_mm_malloc(regions*sizeof(int),16);
    *n_conn_table_G_NMDA = (float *)_mm_malloc(regions*sizeof(float),16);
    *SC_cap = (struct SC_capS *)_mm_malloc(regions*sizeof(struct SC_capS),16);
    SC_capp = *SC_cap;
    *SC_inpreg = (struct SC_inpregS *)_mm_malloc(regions*sizeof(struct SC_inpregS),16);
    SC_inpregp = *SC_inpreg;
    if(*n_conn_table==NULL || *n_conn_table_G_NMDA==NULL || SC_capp==NULL || SC_rowsums==NULL || SC_inpregp==NULL){
        printf("Running out of memory. Terminating.\n");fclose(file_dist);fclose(file_cap);fclose(file_inputreg);exit(2);
    }
    
    // Read the maximal fiber length in header of SCdist-file and compute maxdelay
    if(fscanf(file_dist,"%lf",&tmp) == EOF){
        printf( "ERROR: Unexpected end-of-file in file. File contains less input than expected. Terminating... \n");
        exit(0);
    }
    maxdelay = (int)(((tmp/global_trans_v)*10)+0.5); // *10 for getting from m/s to 10kHz sampling, +0.5 for rounding by casting
    if (maxdelay < 1) maxdelay = 1; // Case: no time delays
    
    // Allocate ringbuffer that contains region activity for each past time-step until maxdelay and another ringbuffer that contains pointers to the first ringbuffer
    *region_activity = (float *)_mm_malloc(maxdelay*regions*sizeof(float),16);
    region_activity_p = *region_activity;
    *reg_globinp_p = (struct Xi_p *)_mm_malloc(maxdelay*regions*sizeof(struct Xi_p),16);
    reg_globinp_pp = *reg_globinp_p;
    if(region_activity_p==NULL || reg_globinp_p==NULL){
        printf("Running out of memory. Terminating.\n");fclose(file_dist);exit(2);
    }
    for (j=0; j<maxdelay*regions; j++) {
        region_activity_p[j]=0.001;
    }
    
    // Read SC files and set pointers for each input region and correspoding delay for each ringbuffer time-step
    int ring_buff_position;
    for (i=0; i<regions; i++) {
        // Read region index of current region (first number of each row) and check whether its consistent for all files
        if(fscanf(file_cap,"%d",&num_incoming_conn_cap) == EOF || fscanf(file_dist,"%d",&num_incoming_conn_dist) == EOF || fscanf(file_inputreg,"%d",&num_incoming_conn_inputreg) == EOF){
            printf( "ERROR: Unexpected end-of-file in SC files. File(s) contain(s) less input than expected. Terminating... \n");
            exit(0);
        }
        if (num_incoming_conn_cap != i || num_incoming_conn_dist != i || num_incoming_conn_inputreg != i) {
            printf( "ERROR: Inconsistencies in global input files, seems like row number is incorrect in some files. i=%d cap=%d dist=%d inp=%d Terminating.\n\n", i, num_incoming_conn_cap, num_incoming_conn_dist, num_incoming_conn_inputreg);
            exit(0);
        }
        
        // Read number of incoming connections for this region (second number of each row) and check whether its consistent across input files
        if(fscanf(file_cap,"%d",&num_incoming_conn_cap) == EOF || fscanf(file_dist,"%d",&num_incoming_conn_dist) == EOF || fscanf(file_inputreg,"%d",&num_incoming_conn_inputreg) == EOF){
            printf( "ERROR: Unexpected end-of-file in file %s or %s. File contains less input than expected. Terminating... \n\n", SC_dist_filename, SC_inputreg_filename);
            exit(0);
        }
        if (num_incoming_conn_cap != num_incoming_conn_inputreg || num_incoming_conn_dist != num_incoming_conn_inputreg) {
            printf( "ERROR: Inconsistencies in SC files: Different numbers of input connections. Terminating.\n\n");
            exit(0);
        }
        
        (*n_conn_table)[i]      = num_incoming_conn_inputreg;
        if (num_incoming_conn_inputreg > 0) {
            (*n_conn_table_G_NMDA)[i]     = G_J_NMDA  / num_incoming_conn_inputreg;
            //(*n_conn_table_G_NMDA)[i]     = G_J_NMDA;
        } else{
            (*n_conn_table_G_NMDA)[i]     = 0;
        }
        
        
        if ((*n_conn_table)[i] > 0) {
            // SC strength and inp region numbers
            SC_capp[i].cap          = (float *)_mm_malloc(((*n_conn_table)[i])*sizeof(float),16);
            SC_inpregp[i].inpreg    = (int *)_mm_malloc(((*n_conn_table)[i])*sizeof(int),16);
            if(SC_capp[i].cap==NULL || SC_inpregp[i].inpreg==NULL){
                printf("Running out of memory. Terminating.\n");exit(2);
            }
            
            // Allocate memory for input-region-pointer arrays for each time-step in ringbuffer
            for (j=0; j<maxdelay; j++){
                reg_globinp_pp[i+j*regions].Xi_elems=(float **)_mm_malloc(((*n_conn_table)[i])*sizeof(float *),16);
                if(reg_globinp_pp[i+j*regions].Xi_elems==NULL){
                    printf("Running out of memory. Terminating.\n");exit(2);
                }
            }
            
            float sum_caps=0.0;
            // Read incoming connections and set pointers
            for (j=0; j<(*n_conn_table)[i]; j++) {
                
                if(fscanf(file_cap,"%lf",&tmp) != EOF && fscanf(file_dist,"%lf",&tmp2) != EOF && fscanf(file_inputreg,"%d",&tmp3) != EOF){
                    
                    tmpint = (int)(((tmp2/global_trans_v)*10)+0.5); //  *10 for getting from m/s or mm/ms to 10kHz sampling, +0.5 for rounding by casting
                    if (tmpint < 0 || tmpint > maxdelay){
                        printf( "\nERROR: Negative or too high (larger than maximum specified number) connection length/delay %d -> %d. Terminating... \n\n",i,tmp3);exit(0);
                    }
                    if (tmpint == 0) tmpint = 1; // If time delay is smaller than integration step size, than set time delay to one integration step
                    
                    //SC_capp[i].cap[j] = (float)tmp * (*n_conn_table_G_NMDA)[i];
                    SC_capp[i].cap[j] = (float)tmp * G_J_NMDA;
                    //sum_caps                += (float)tmp;
                    sum_caps                += SC_capp[i].cap[j];
                    SC_inpregp[i].inpreg[j]  =  tmp3;
                    
                    if (tmp3 >= 0 && tmp3 < regions) {
                        ring_buff_position=maxdelay*regions - tmpint*regions + tmp3;
                        for (k=0; k<maxdelay; k++) {
                            reg_globinp_pp[i+k*regions].Xi_elems[j]=&region_activity_p[ring_buff_position];
                            ring_buff_position += regions;
                            if (ring_buff_position > (maxdelay*regions-1)) ring_buff_position -= maxdelay*regions;
                        }
                    } else {
                        printf( "\nERROR: Region index is negative or too large: %d -> %d. Terminating... \n\n",i,tmp3);exit(0);
                    }
                    
                    
                } else{
                    printf( "\nERROR: Unexpected end-of-file in file %s or %s. File contains less input than expected. Terminating... \n\n", SC_inputreg_filename, SC_dist_filename);
                    exit(0);
                }
                
            }
            if (sum_caps <= 0) {
                printf( "\nERROR: Sum of connection strenghts is negative or zero. sum-caps node %d = %f. Terminating... \n\n",i,sum_caps);exit(0);
            }
            (*SC_rowsums)[i] = sum_caps;
            /*
             for (j=0; j<(*n_conn_table)[i]; j++) {
             SC_capp[i].cap[j] /= sum_caps;
             }
             */
        }
    }
    
    fclose(file_dist);fclose(file_inputreg);
    return maxdelay;
}


/*
 Usage: tvbii <paramfile> <subject_id>
 */

int main(int argc, char *argv[])
{
    /*
     Get current time, initialize random number generator with seed
     */
    time_t start = time(NULL);
    
    int i, j, k;
    
    
    /*
     Open input/output file(s)
     */
    if (argc != 3) {
        printf( "\nERROR: Wrong number of arguments.\n\nUsage: tvbii <paramfile> <subid>\n\nTerminating... \n\n");
        exit(0);
    }
    openFCoutfile(argv[1]);
    char subject_file[120];memset(subject_file, 0, 120*sizeof(char));
    strcpy(subject_file,"input/");strcat(subject_file,argv[2]);
    strcat(subject_file,"_input_exc.txt");
    char subject_file2[120];memset(subject_file2, 0, 120*sizeof(char));
    strcpy(subject_file2,"input/");strcat(subject_file2,argv[2]);
    strcat(subject_file2,"_input_inh.txt");
    
    
    
    
    /*
     Global model and integration parameters
     */
    const float dt                  = 0.1;      // Integration step length dt = 0.1 ms
    const float model_dt            = 0.001;    // Time-step of model (sampling-rate=1000 Hz)
    const int   vectorization_grade = 4;        // How many operations can be done simultaneously. Depends on CPU Architecture and available intrinsics.
    int         time_steps          = 667*1.94*1000;    // Simulation length
    int         FIC_iters           = 100;                                               // Number of FIC iterations
    int         FIC_time_steps      = 10 * 1000;                                        // Length of FIC simulations (default: 10 s)
    int         FIC_burn_in_ts      = 2  * 1000;
    int         nodes               = 84;    // Number of surface vertices; must be a multiple of vectorization grade
    int         regions             = 84;    // Number of large-scale regions
    float       global_trans_v      = 1.0;     // Global transmission velocity (m/s); Local time-delays can be ommited since smaller than integration time-step
    float       G                   = 0.5;        // Global coupling strength
    int         BOLD_TR             = 1940;     // TR of BOLD data
    /*
     Local model: DMF-Parameters from Deco et al. JNeuro 2014
     */
    float w_plus  = 1.4;          // local excitatory recurrence
//    float I_ext   = 0;            // External stimulation
    float J_NMDA  = 0.15;         // (nA) excitatory synaptic coupling
    //float J_i     = 1.0;          //
    const float a_E     = 310;          // (n/C)
    const float b_E     = 125;          // (Hz)
    const float d_E     = 0.16;         // (s)
    const float a_I     = 615;          // (n/C)
    const float b_I     = 177;          // (Hz)
    const float d_I     = 0.087;        // (s)
    const float gamma   = 0.641/1000.0; // factor 1000 for expressing everything in ms
    const float tau_E   = 100;          // (ms) Time constant of NMDA (excitatory)
    const float tau_I   = 10;           // (ms) Time constant of GABA (inhibitory)
    float       sigma   = 0.00316228;   // (nA) Noise amplitude
    const float I_0     = 0.382;        // (nA) overall effective external input
    const float w_E     = 1.0;          // scaling of external input for excitatory pool
    const float w_I     = 0.7;          // scaling of external input for inhibitory pool
    const float gamma_I = 1.0/1000.0;   // for expressing inhib. pop. in ms
    float       tmpJi   = 0.0;          // Feedback inhibition J_i
    
    /*
     Input-file mode is on: overwrite some parameter-values as specified in additional param-file
     */
    FILE *file;
    file=fopen(argv[1], "r");
    int rand_num_seed = 1403;
    if (file==NULL){
        printf( "\nERROR: Could not open file %s. Terminating... \n\n", argv[1]);
        exit(0);
    }
    // paramfile: <param1> <param2> ...
    if(fscanf(file,"%d",&nodes) != EOF && fscanf(file,"%f",&G) != EOF && fscanf(file,"%f",&J_NMDA) != EOF && fscanf(file,"%f",&w_plus) != EOF && fscanf(file,"%f",&tmpJi) != EOF && fscanf(file,"%f",&sigma) != EOF && fscanf(file,"%d",&time_steps) != EOF && fscanf(file,"%d",&FIC_time_steps) != EOF && fscanf(file,"%d",&BOLD_TR) != EOF && fscanf(file,"%f",&global_trans_v) != EOF && fscanf(file,"%d",&rand_num_seed) != EOF){
    } else{
        printf( "\nERROR: Unexpected end-of-file in file %s. File contains less input than expected. Terminating... \n\n", argv[1]);
        exit(0);
    }
    fclose(file);
    
    if (nodes % vectorization_grade != 0){
        printf( "\nERROR: Specified number of nodes (%d) is not a multiple of vectorization grade (%d). Terminating... \n\n", nodes, vectorization_grade);
        exit(0);
    }
    regions = nodes;
    srand((unsigned)rand_num_seed);
    
    
    /*
     Allocate and Initialize arrays
     */
    float *S_i_E                    = (float *)_mm_malloc(nodes * sizeof(float),16);
    float *S_i_I                    = (float *)_mm_malloc(nodes * sizeof(float),16);
    float *r_i_E                    = (float *)_mm_malloc(nodes * sizeof(float),16);
    float *r_i_I                    = (float *)_mm_malloc(nodes * sizeof(float),16);
    
    
    float *global_input             = (float *)_mm_malloc(nodes * sizeof(float),16);
    float *J_i                      = (float *)_mm_malloc(nodes * sizeof(float),16);  // (nA) inhibitory synaptic coupling
    float *meanFR                   = (float *)_mm_malloc(nodes * sizeof(float),16);  // summation array for mean firing rate
    if (S_i_E == NULL || S_i_I == NULL || r_i_E == NULL || r_i_I == NULL ||  global_input == NULL || J_i == NULL || meanFR == NULL) {
        printf( "ERROR: Running out of memory. Aborting... \n");
        _mm_free(S_i_E);_mm_free(S_i_I);_mm_free(global_input);_mm_free(J_i);_mm_free(meanFR);
        return 1;
    }
    //Balloon-Windkessel model arrays
    float TR        = (float)BOLD_TR / 1000;                                                             // (s) TR of fMRI data
    int   output_ts = time_steps / (TR / model_dt);                                     // Length of BOLD time-series written to HDD
    int   num_output_ts      = nodes;                                                    // Number of BOLD time-series that are writte to HDD
    float *bw_x_ex    = (float *)_mm_malloc(num_output_ts * sizeof(float),16);             // State-variable 1 of BW-model (exc. pop.)
    float *bw_f_ex    = (float *)_mm_malloc(num_output_ts * sizeof(float),16);             // State-variable 2 of BW-model (exc. pop.)
    float *bw_nu_ex   = (float *)_mm_malloc(num_output_ts * sizeof(float),16);             // State-variable 3 of BW-model (exc. pop.)
    float *bw_q_ex    = (float *)_mm_malloc(num_output_ts * sizeof(float),16);             // State-variable 4 of BW-model (exc. pop.)
    float *bw_x_in    = (float *)_mm_malloc(num_output_ts * sizeof(float),16);             // State-variable 1 of BW-model (inh. pop.)
    float *bw_f_in    = (float *)_mm_malloc(num_output_ts * sizeof(float),16);             // State-variable 2 of BW-model (inh. pop.)
    float *bw_nu_in   = (float *)_mm_malloc(num_output_ts * sizeof(float),16);             // State-variable 3 of BW-model (inh. pop.)
    float *bw_q_in    = (float *)_mm_malloc(num_output_ts * sizeof(float),16);             // State-variable 4 of BW-model (inh. pop.)
    
  
    
    // Derived parameters
    const int   nodes_vec     = nodes/vectorization_grade;
    const float meanFRfact    = 1.0 / (time_steps*10);
    const float min_d_E       = -1.0 * d_E;
    const float min_d_I       = -1.0 * d_I;
    const float imintau_E     = -1.0 / tau_E;
    const float imintau_I     = -1.0 / tau_I;
    const float w_E__I_0      = w_E * I_0;
    const float w_I__I_0      = w_I * I_0;
    const float one           = 1.0;
    const float w_plus__J_NMDA= w_plus * J_NMDA;
    const float G_J_NMDA      = G * J_NMDA;
    float mean_mean_FR=0.0;
    
    // Initialize state variables / parameters
    for (j = 0; j < nodes; j++) {
        S_i_E[j]            = 0.001;
        S_i_I[j]            = 0.001;
        global_input[j]     = 0.001;
        meanFR[j]           = 0.0f;
    }
    
    float       tmpglobinput;
    int         ring_buf_pos=0;
    float tmp_exp_E[4]          __attribute__((aligned(16)));
    float tmp_exp_I[4]          __attribute__((aligned(16)));
    float rand_number[4]        __attribute__((aligned(16)));
    
    /*
     Import and setup global and local connectivity
     */
    int         *n_conn_table;
    float       *region_activity, *n_conn_table_G_NMDA, *SC_rowsums;
    struct Xi_p *reg_globinp_p;
    struct SC_capS      *SC_cap;
    struct SC_inpregS   *SC_inpreg;
    char cap_file[100];memset(cap_file, 0, 100*sizeof(char));
    strcpy(cap_file,"input/");strcat(cap_file,argv[2]);strcat(cap_file,"_SC_strengths.txt");
    char dist_file[100];memset(dist_file, 0, 100*sizeof(char));
    strcpy(dist_file,"input/");strcat(dist_file,argv[2]);strcat(dist_file,"_SC_distances.txt");
    char reg_file[100];memset(reg_file, 0, 100*sizeof(char));
    strcpy(reg_file,"input/");strcat(reg_file,argv[2]);strcat(reg_file,"_SC_regionids.txt");
    
    int         maxdelay = importGlobalConnectivity(cap_file, dist_file, reg_file, regions, &region_activity, &reg_globinp_p, global_trans_v, &n_conn_table, &n_conn_table_G_NMDA, G_J_NMDA, &SC_cap, &SC_rowsums, &SC_inpreg);
    int         reg_act_size = regions * maxdelay;
    
    // That's a first guess for good J_i values
    for (j = 0; j < nodes; j++) {
        J_i[j]              = 1.0 + 4.0 * SC_rowsums[j] ;
    }
    
    
    
    /*
     Initialize or cast to vector-intrinsics types for variables & parameters
     */
    const __m128    _dt                 = _mm_load1_ps(&dt);
    const __m128    _w_plus_J_NMDA      = _mm_load1_ps(&w_plus__J_NMDA);
    const __m128    _a_E                = _mm_load1_ps(&a_E);
    const __m128    _b_E                = _mm_load1_ps(&b_E);
    const __m128    _min_d_E            = _mm_load1_ps(&min_d_E);
    const __m128    _a_I                = _mm_load1_ps(&a_I);
    const __m128    _b_I                = _mm_load1_ps(&b_I);
    const __m128    _min_d_I            = _mm_load1_ps(&min_d_I);
    const __m128    _gamma              = _mm_load1_ps(&gamma);
    const __m128    _gamma_I            = _mm_load1_ps(&gamma_I);
    const __m128    _imintau_E          = _mm_load1_ps(&imintau_E);
    const __m128    _imintau_I          = _mm_load1_ps(&imintau_I);
    const __m128    _w_E__I_0           = _mm_load1_ps(&w_E__I_0);
    const __m128    _w_I__I_0           = _mm_load1_ps(&w_I__I_0);
    float           tmp_sigma           = sigma*dt;// pre-compute dt*sigma for the integration of sigma*randnumber in equations (9) and (10) of Deco2014
    const __m128    _sigma              = _mm_load1_ps(&tmp_sigma);
    //const __m128    _I_0                = _mm_load1_ps(&I_0);
    const __m128    _one                = _mm_load1_ps(&one);
    const __m128    _J_NMDA             = _mm_load1_ps(&J_NMDA);

    
    __m128          *_S_i_E             = (__m128*)S_i_E;
    __m128          *_S_i_I             = (__m128*)S_i_I;
    __m128          *_r_i_E             = (__m128*)r_i_E;
    __m128          *_r_i_I             = (__m128*)r_i_I;
    
    
    __m128          *_tmp_exp_E         = (__m128*)tmp_exp_E;
    __m128          *_tmp_exp_I         = (__m128*)tmp_exp_I;
    __m128          *_rand_number       = (__m128*)rand_number;
    __m128          *_global_input      = (__m128*)global_input;
    __m128          *_J_i               = (__m128*)J_i;
    __m128          *_meanFR            = (__m128*)meanFR;
    __m128          _tmp_I_E, _tmp_I_I;
    __m128          _tmp_H_E, _tmp_H_I;
    float BOLD_ex[num_output_ts][output_ts+2];
    float rho = 0.34, alpha = 0.32, tau = 0.98, y = 1.0/0.41, kappa = 1.0/0.65;
    float V_0 = 0.02, k1 = 7 * rho, k2 = 2.0, k3 = 2 * rho - 0.2, ialpha = 1.0/alpha, itau = 1.0/tau, oneminrho = (1.0 - rho);
    float f_tmp;
    int   BOLD_len_i  = -1;
    int ts_bold=0,ts, int_i, i_node_vec, ext_inp_counter=0;
    
    

    
    /*
     Parameters for FIC tuning
     */
    float           *FIC_I_E            = (float *)_mm_malloc(nodes * sizeof(float),16);    // Average input to excitatory population during FIC tuning
    __m128          *_FIC_I_E           = (__m128*)FIC_I_E;
    float           *FIC_delta          = (float *)_mm_malloc(nodes * sizeof(float),16);    // Change by which node-wise J_i is up-/down-regulated in each FIC iter
    float           *best_Ji            = (float *)_mm_malloc(nodes * sizeof(float),16);    // Change by which node-wise J_i is up-/down-regulated in each FIC iter
    float           *tmp_Ji             = (float *)_mm_malloc(nodes * sizeof(float),16);    // Change by which node-wise J_i is up-/down-regulated in each FIC iter
    
    const float     FIC_norm_fact       = (float)(FIC_time_steps - FIC_burn_in_ts) * 10.0;  // Factor for computing mean I_E for FIC
    const float     FIC_norm_term       = FIC_norm_fact * 125.0 / 310.0;                    // Additive term for computing mean I_E for FIC
    int             i_fic, FIC_termination_flag = 0, max_num_nodes = 0, best_iter = 0;
    float           tuning_factor       = 0.02, best_mean_FR = 0.0;
    
    for (j = 0; j < nodes; j++) {     // Initialize arrays
        FIC_delta[j]                    = 0.02;
        best_Ji[j]                      = J_i[j];
        tmp_Ji[j]                       = J_i[j];
    }
    
    /*
     Parameters for Connectome tuning
     */
    int             SCtune_iters        = 1;
    int             i_SC;
    
    
    /*
     Connectome-tuning loop
     */
    for (i_SC = 0; i_SC < SCtune_iters; i_SC++) {
        
        
        
        /*
         ******************************************************************************************
         *************************************** FIC TUNING ***************************************
         ******************************************************************************************
         */
        
        tuning_factor       = 1.0;
        max_num_nodes       = 0;
        best_mean_FR        = 0;
        for (i_fic = 0; i_fic < FIC_iters; i_fic++) {
            
            // Reset arrays
            for (j = 0; j < nodes; j++) {
                S_i_E[j]            = 0.001;
                S_i_I[j]            = 0.001;
                global_input[j]     = 0.001;
                meanFR[j]           = 0.0;
                FIC_I_E[j]          = 0.0;
            }
            ring_buf_pos        = 0;
            for (j=0; j<maxdelay*regions; j++) {
                region_activity[j]=0.001;
            }
            
            
            /*
             Burn-in phase
             */
            for (ts = 0; ts < FIC_burn_in_ts; ts++) {
                for (int_i = 0; int_i < 10; int_i++) {
                    
                    /*
                     Compute global coupling
                     */
                    for(j=0; j<regions; j++){
                        tmpglobinput = 0;
                        for (k=0; k<n_conn_table[j]; k++) {
                            tmpglobinput += *reg_globinp_p[j+ring_buf_pos].Xi_elems[k] * SC_cap[j].cap[k];
                        }
                        global_input[j] = tmpglobinput;
                    }
                    
                    for (i_node_vec = 0; i_node_vec < nodes_vec; i_node_vec++) {
                        // Excitatory population firing rate
                        _tmp_I_E    = _mm_sub_ps(_mm_mul_ps(_a_E,_mm_add_ps(_mm_add_ps(_w_E__I_0,_mm_mul_ps(_w_plus_J_NMDA, _S_i_E[i_node_vec])),_mm_sub_ps(_global_input[i_node_vec],_mm_mul_ps(_J_i[i_node_vec], _S_i_I[i_node_vec])))),_b_E);
                        
                        *_tmp_exp_E     = _mm_mul_ps(_min_d_E, _tmp_I_E);
                        tmp_exp_E[0]    = tmp_exp_E[0] != 0 ? expf(tmp_exp_E[0]) : 0.9;
                        tmp_exp_E[1]    = tmp_exp_E[1] != 0 ? expf(tmp_exp_E[1]) : 0.9;
                        tmp_exp_E[2]    = tmp_exp_E[2] != 0 ? expf(tmp_exp_E[2]) : 0.9;
                        tmp_exp_E[3]    = tmp_exp_E[3] != 0 ? expf(tmp_exp_E[3]) : 0.9;
                        _tmp_H_E        = _mm_div_ps(_tmp_I_E, _mm_sub_ps(_one, *_tmp_exp_E));
                        
                        //_meanFR[i_node_vec] = _mm_add_ps(_meanFR[i_node_vec],_tmp_H_E);
                        _r_i_E[i_node_vec]  = _tmp_H_E;
                        
                        // Inhibitory population firing rate
                        _tmp_I_I = _mm_sub_ps(_mm_mul_ps(_a_I,_mm_sub_ps(_mm_add_ps(_w_I__I_0,_mm_mul_ps(_J_NMDA, _S_i_E[i_node_vec])), _S_i_I[i_node_vec])),_b_I);
                        *_tmp_exp_I   = _mm_mul_ps(_min_d_I, _tmp_I_I);
                        tmp_exp_I[0]  = tmp_exp_I[0] != 0 ? expf(tmp_exp_I[0]) : 0.9;
                        tmp_exp_I[1]  = tmp_exp_I[1] != 0 ? expf(tmp_exp_I[1]) : 0.9;
                        tmp_exp_I[2]  = tmp_exp_I[2] != 0 ? expf(tmp_exp_I[2]) : 0.9;
                        tmp_exp_I[3]  = tmp_exp_I[3] != 0 ? expf(tmp_exp_I[3]) : 0.9;
                        _tmp_H_I  = _mm_div_ps(_tmp_I_I, _mm_sub_ps(_one, *_tmp_exp_I));
                        _r_i_I[i_node_vec] = _tmp_H_I;
                        
                        
                        gaussrand(rand_number);
                        _S_i_I[i_node_vec] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_sigma, *_rand_number),_S_i_I[i_node_vec]),_mm_mul_ps(_dt,_mm_add_ps(_mm_mul_ps(_imintau_I, _S_i_I[i_node_vec]),_mm_mul_ps(_tmp_H_I,_gamma_I))));
                        
                        gaussrand(rand_number);
                        _S_i_E[i_node_vec] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_sigma, *_rand_number),_S_i_E[i_node_vec]),_mm_mul_ps(_dt, _mm_add_ps(_mm_mul_ps(_imintau_E, _S_i_E[i_node_vec]),_mm_mul_ps(_mm_mul_ps(_mm_sub_ps(_one, _S_i_E[i_node_vec]),_gamma),_tmp_H_E))));
                    }
                    memcpy(&region_activity[ring_buf_pos], S_i_E, regions*sizeof( float ));
                    ring_buf_pos = ring_buf_pos<(reg_act_size-regions) ? (ring_buf_pos+regions) : 0;
                }
            }
            
            /*
             Main FIC simulation phase
             */
            for ( ; ts < FIC_time_steps; ts++) {
                for (int_i = 0; int_i < 10; int_i++) {
                    
                    /*
                     Compute global coupling
                     */
                    for(j=0; j<regions; j++){
                        tmpglobinput = 0;
                        for (k=0; k<n_conn_table[j]; k++) {
                            tmpglobinput += *reg_globinp_p[j+ring_buf_pos].Xi_elems[k] * SC_cap[j].cap[k];
                        }
                        global_input[j] = tmpglobinput;
                    }
                    
                    for (i_node_vec = 0; i_node_vec < nodes_vec; i_node_vec++) {
                        // Excitatory population firing rate
                        //_tmp_I_E    = _mm_sub_ps(_mm_mul_ps(_a_E,_mm_add_ps(_mm_add_ps(_w_E__I_0,_mm_mul_ps(_w_plus_J_NMDA, _S_i_E[i_node_vec])),_mm_sub_ps(_global_input[i_node_vec],_mm_mul_ps(_J_i[i_node_vec], _S_i_I[i_node_vec])))),_b_E);
                        
                        _tmp_I_E        = _mm_add_ps(_mm_add_ps(_w_E__I_0,_mm_mul_ps(_w_plus_J_NMDA, _S_i_E[i_node_vec])),_mm_sub_ps(_global_input[i_node_vec],_mm_mul_ps(_J_i[i_node_vec], _S_i_I[i_node_vec])));
                        _FIC_I_E[i_node_vec] = _mm_add_ps(_FIC_I_E[i_node_vec],_tmp_I_E);
                        _tmp_I_E        = _mm_sub_ps(_mm_mul_ps(_a_E,_tmp_I_E),_b_E);
                        *_tmp_exp_E     = _mm_mul_ps(_min_d_E, _tmp_I_E);
                        tmp_exp_E[0]    = tmp_exp_E[0] != 0 ? expf(tmp_exp_E[0]) : 0.9;
                        tmp_exp_E[1]    = tmp_exp_E[1] != 0 ? expf(tmp_exp_E[1]) : 0.9;
                        tmp_exp_E[2]    = tmp_exp_E[2] != 0 ? expf(tmp_exp_E[2]) : 0.9;
                        tmp_exp_E[3]    = tmp_exp_E[3] != 0 ? expf(tmp_exp_E[3]) : 0.9;
                        _tmp_H_E        = _mm_div_ps(_tmp_I_E, _mm_sub_ps(_one, *_tmp_exp_E));
                        
                        _meanFR[i_node_vec] = _mm_add_ps(_meanFR[i_node_vec],_tmp_H_E);
                        _r_i_E[i_node_vec]  = _tmp_H_E;
                        
                        // Inhibitory population firing rate
                        _tmp_I_I = _mm_sub_ps(_mm_mul_ps(_a_I,_mm_sub_ps(_mm_add_ps(_w_I__I_0,_mm_mul_ps(_J_NMDA, _S_i_E[i_node_vec])), _S_i_I[i_node_vec])),_b_I);
                        *_tmp_exp_I   = _mm_mul_ps(_min_d_I, _tmp_I_I);
                        tmp_exp_I[0]  = tmp_exp_I[0] != 0 ? expf(tmp_exp_I[0]) : 0.9;
                        tmp_exp_I[1]  = tmp_exp_I[1] != 0 ? expf(tmp_exp_I[1]) : 0.9;
                        tmp_exp_I[2]  = tmp_exp_I[2] != 0 ? expf(tmp_exp_I[2]) : 0.9;
                        tmp_exp_I[3]  = tmp_exp_I[3] != 0 ? expf(tmp_exp_I[3]) : 0.9;
                        _tmp_H_I  = _mm_div_ps(_tmp_I_I, _mm_sub_ps(_one, *_tmp_exp_I));
                        _r_i_I[i_node_vec] = _tmp_H_I;
                        
                        
                        gaussrand(rand_number);
                        _S_i_I[i_node_vec] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_sigma, *_rand_number),_S_i_I[i_node_vec]),_mm_mul_ps(_dt,_mm_add_ps(_mm_mul_ps(_imintau_I, _S_i_I[i_node_vec]),_mm_mul_ps(_tmp_H_I,_gamma_I))));
                        
                        gaussrand(rand_number);
                        _S_i_E[i_node_vec] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_sigma, *_rand_number),_S_i_E[i_node_vec]),_mm_mul_ps(_dt, _mm_add_ps(_mm_mul_ps(_imintau_E, _S_i_E[i_node_vec]),_mm_mul_ps(_mm_mul_ps(_mm_sub_ps(_one, _S_i_E[i_node_vec]),_gamma),_tmp_H_E))));
                    }
                    memcpy(&region_activity[ring_buf_pos], S_i_E, regions*sizeof( float ));
                    ring_buf_pos = ring_buf_pos<(reg_act_size-regions) ? (ring_buf_pos+regions) : 0;
                }
            }
            
            
            
            tuning_factor *= 0.96;
            if (tuning_factor < 0.4) tuning_factor = 0.4;
            
            
            FIC_termination_flag = 0;
            mean_mean_FR = 0;
            printf("*****************************************************************\n");
            for (j = 0; j < nodes; j++){
                tmp_Ji[j]     = J_i[j];
                FIC_I_E[j]    = (FIC_I_E[j] - FIC_norm_term) / FIC_norm_fact;
                meanFR[j]     = meanFR[j] / FIC_norm_fact;
                mean_mean_FR  += meanFR[j];
                if ( fabsf(FIC_I_E[j] + 0.026f) >  0.005) {
                    
                    FIC_delta[j]  =   (0.9253 * powf(meanFR[j], 0.1337) - 1.08) * tuning_factor;
                    //FIC_delta[j]  =  (meanFR[j] - 3.0631) * tuning_factor;
                    
                    J_i[j]       +=  FIC_delta[j];
                    
                    /*
                     if (FIC_I_E[j] < -0.026) {
                     J_i[j]       -= FIC_delta[j];
                     FIC_delta[j] -= 0.001;
                     if (FIC_delta[j] < 0.001) {
                     FIC_delta[j] = 0.001;
                     }
                     } else {
                     J_i[j]       += FIC_delta[j];
                     }
                     */
                } else {
                    FIC_termination_flag++;
                    FIC_delta[j]  =   0.0;
                }
                printf("%d ### %.4f \t\t %.2f \t\t %.3f \t\t %.2f \n", j, FIC_I_E[j], J_i[j], FIC_delta[j], meanFR[j]);
            }
            mean_mean_FR  /= nodes;
            
            
            // Current tuning is current optimum, so store these settings
            if (FIC_termination_flag >= max_num_nodes) {
                max_num_nodes = FIC_termination_flag;
                best_mean_FR  = mean_mean_FR;
                best_iter     = i_fic;
                
                for (j = 0; j < nodes; j++){
                    best_Ji[j]                      = tmp_Ji[j];
                }
            }
            
            printf("Iter: %d   N_nodes: %d / %d (%.2f Hz, iter: %d)   tuningfact: %.5f    meanFR: %.2f\n", i_fic, FIC_termination_flag,max_num_nodes,best_mean_FR, best_iter, tuning_factor,mean_mean_FR);
            
            // This is the goal of FIC tuning
            if (FIC_termination_flag == nodes) {
                break;
            }
            
        }
        
        
        /*
         Write out J_i values found during FIC tuning
         */
        for (j=0; j<nodes; j++) {
            fprintf(FCout, "%.7f %.7f\n",SC_rowsums[j],best_Ji[j]);
        }

    
        // Use J_i values that gave best results during FIC tuning
        for (j = 0; j < nodes; j++){
            J_i[j]          =       best_Ji[j];
        }
        


        
        /*
         *************************************************************************************************
         *************************************** END OF FIC TUNING ***************************************
         *************************************************************************************************
         */
        
        
        
        
        /*
         Reset arrays
         */
        for (j = 0; j < nodes; j++) {
            S_i_E[j]            = 0.001;
            S_i_I[j]            = 0.001;
            global_input[j]     = 0.001;
            meanFR[j]           = 0.0;
        }
        ring_buf_pos        = 0;
        for (j=0; j<maxdelay*regions; j++) {
            region_activity[j]=0.001;
        }

        
        /*
         Reset Balloon-Windkessel model parameters and arrays
         */
        for (j = 0; j < num_output_ts; j++) {
            bw_x_ex[j] = 0.0;
            bw_f_ex[j] = 1.0;
            bw_nu_ex[j] = 1.0;
            bw_q_ex[j] = 1.0;
            bw_x_in[j] = 0.0;
            bw_f_in[j] = 1.0;
            bw_nu_in[j] = 1.0;
            bw_q_in[j] = 1.0;
        }
        
        
        /*
         The simulation starts
         */
        ts_bold         =   0;
        BOLD_len_i      =   -1;
        ext_inp_counter =   0;
        for (ts = 0; ts < time_steps; ts++) {
            printf("%.1f %% \r", ((float)ts / (float)time_steps) * 100.0f );

            for (int_i = 0; int_i < 10; int_i++) {
                /*
                 Compute global coupling
                 */
                // 1. Time-delayed and capacity weighted long-range input for next time-step
                for(j=0; j<regions; j++){
                    tmpglobinput = 0;
                    for (k=0; k<n_conn_table[j]; k++) {
                        tmpglobinput += *reg_globinp_p[j+ring_buf_pos].Xi_elems[k] * SC_cap[j].cap[k];
                    }
                    
                    // CAUTION: Change this if #regions != #nodes
                    //global_input_reg[j] = tmpglobinput * n_conn_table_G_NMDA[j];
                    //global_input[j] = tmpglobinput * n_conn_table_G_NMDA[j];
                    global_input[j] = tmpglobinput;
                }
                
                for (i_node_vec = 0; i_node_vec < nodes_vec; i_node_vec++) {
                    
                    // Excitatory population firing rate
                    _tmp_I_E    = _mm_sub_ps(_mm_mul_ps(_a_E,_mm_add_ps(_mm_add_ps(_w_E__I_0,_mm_mul_ps(_w_plus_J_NMDA, _S_i_E[i_node_vec])),_mm_sub_ps(_global_input[i_node_vec],_mm_mul_ps(_J_i[i_node_vec], _S_i_I[i_node_vec])))),_b_E);
                    *_tmp_exp_E   = _mm_mul_ps(_min_d_E, _tmp_I_E);
                    tmp_exp_E[0]  = tmp_exp_E[0] != 0 ? expf(tmp_exp_E[0]) : 0.9;
                    tmp_exp_E[1]  = tmp_exp_E[1] != 0 ? expf(tmp_exp_E[1]) : 0.9;
                    tmp_exp_E[2]  = tmp_exp_E[2] != 0 ? expf(tmp_exp_E[2]) : 0.9;
                    tmp_exp_E[3]  = tmp_exp_E[3] != 0 ? expf(tmp_exp_E[3]) : 0.9;
                    _tmp_H_E  = _mm_div_ps(_tmp_I_E, _mm_sub_ps(_one, *_tmp_exp_E));
                    
                    _meanFR[i_node_vec] = _mm_add_ps(_meanFR[i_node_vec],_tmp_H_E);
                    _r_i_E[i_node_vec] = _tmp_H_E;
                    
                    // Inhibitory population firing rate
                    
                    _tmp_I_I = _mm_sub_ps(_mm_mul_ps(_a_I,_mm_sub_ps(_mm_add_ps(_w_I__I_0,_mm_mul_ps(_J_NMDA, _S_i_E[i_node_vec])), _S_i_I[i_node_vec])),_b_I);
                    *_tmp_exp_I   = _mm_mul_ps(_min_d_I, _tmp_I_I);
                    tmp_exp_I[0]  = tmp_exp_I[0] != 0 ? expf(tmp_exp_I[0]) : 0.9;
                    tmp_exp_I[1]  = tmp_exp_I[1] != 0 ? expf(tmp_exp_I[1]) : 0.9;
                    tmp_exp_I[2]  = tmp_exp_I[2] != 0 ? expf(tmp_exp_I[2]) : 0.9;
                    tmp_exp_I[3]  = tmp_exp_I[3] != 0 ? expf(tmp_exp_I[3]) : 0.9;
                    _tmp_H_I  = _mm_div_ps(_tmp_I_I, _mm_sub_ps(_one, *_tmp_exp_I));
                    _r_i_I[i_node_vec] = _tmp_H_I;
                    
                    // Compute synaptic activity
                    // CAUTION: In these equations dt * sigma is pre-computed above
                    
                    gaussrand(rand_number);
                    _S_i_I[i_node_vec] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_sigma, *_rand_number),_S_i_I[i_node_vec]),_mm_mul_ps(_dt,_mm_add_ps(_mm_mul_ps(_imintau_I, _S_i_I[i_node_vec]),_mm_mul_ps(_tmp_H_I,_gamma_I))));
                    
                    gaussrand(rand_number);
                    _S_i_E[i_node_vec] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_sigma, *_rand_number),_S_i_E[i_node_vec]),_mm_mul_ps(_dt, _mm_add_ps(_mm_mul_ps(_imintau_E, _S_i_E[i_node_vec]),_mm_mul_ps(_mm_mul_ps(_mm_sub_ps(_one, _S_i_E[i_node_vec]),_gamma),_tmp_H_E))));
                    
                }
                //ext_inp_counter -= nodes_vec;
                memcpy(&region_activity[ring_buf_pos], S_i_E, regions*sizeof( float ));
                // 3. Shift ring-buff-pos
                ring_buf_pos = ring_buf_pos<(reg_act_size-regions) ? (ring_buf_pos+regions) : 0;
            }
            
            
            
            /*
             Compute BOLD for that time-step (subsampled to 1 ms)
             */
            
            for (j = 0; j < num_output_ts; j++) {
                bw_x_ex[j]  = bw_x_ex[j]  +  model_dt * (S_i_E[j] - kappa * bw_x_ex[j] - y * (bw_f_ex[j] - 1.0));
                f_tmp       = bw_f_ex[j]  +  model_dt * bw_x_ex[j];
                bw_nu_ex[j] = bw_nu_ex[j] +  model_dt * itau * (bw_f_ex[j] - powf(bw_nu_ex[j], ialpha));
                bw_q_ex[j]  = bw_q_ex[j]  +  model_dt * itau * (bw_f_ex[j] * (1.0 - powf(oneminrho,(1.0/bw_f_ex[j]))) / rho  - powf(bw_nu_ex[j],ialpha) * bw_q_ex[j] / bw_nu_ex[j]);
                bw_f_ex[j]  = f_tmp;
                
                /*
                 bw_x_in[j]  = bw_x_in[j]  +  model_dt * (S_i_I[j] - kappa * bw_x_in[j] - y * (bw_f_in[j] - 1.0));
                 f_tmp       = bw_f_in[j]  +  model_dt * bw_x_in[j];
                 bw_nu_in[j] = bw_nu_in[j] +  model_dt * itau * (bw_f_in[j] - powf(bw_nu_in[j], ialpha));
                 bw_q_in[j]  = bw_q_in[j]  +  model_dt * itau * (bw_f_in[j] * (1.0 - powf(oneminrho,(1.0/bw_f_in[j]))) / rho  - powf(bw_nu_in[j],ialpha) * bw_q_in[j] / bw_nu_in[j]);
                 bw_f_in[j]  = f_tmp;
                 */
            }
            
            /*
             
             for (j = 0; j < num_output_ts; j++) {
             fprintf(WFout, "%.4f ",S_i_E[j]);
             }
             
             
             
             for (j = 0; j < num_output_ts; j++) {
             fprintf(WFout, "%.4f ",S_i_I[j]);
             }
             for (j = 0; j < num_output_ts; j++) {
             fprintf(WFout, "%.2f ",r_i_E[j]);
             }
             
             for (j = 0; j < num_output_ts; j++) {
             fprintf(WFout, "%.2f ",r_i_I[j]);
             }
             
             fprintf(WFout, "\n");
             */
            
            if (ts_bold % BOLD_TR == 0) {
                //printf( "ts: %d\n", ts);
                BOLD_len_i++;
                
                for (j = 0; j < num_output_ts; j++) {
                    BOLD_ex[j][BOLD_len_i] = 100 / rho * V_0 * (k1 * (1 - bw_q_ex[j]) + k2 * (1 - bw_q_ex[j]/bw_nu_ex[j]) + k3 * (1 - bw_nu_ex[j]));
                    //BOLD_in[BOLD_len_i * num_output_ts + j] =  100 / rho * V_0 * (k1 * (1 - bw_q_ex[j]) + k2 * (1 - bw_q_ex[j]/bw_nu_ex[j]) + k3 * (1 - bw_nu_ex[j]));
                }
            }
            ts_bold++;
            
        } // Simulation loop
        
        
        /*
         Compute mean firing rate
         */
        mean_mean_FR = 0;
        for (j = 0; j < nodes; j++){
            meanFR[j] = meanFR[j]*meanFRfact;
            mean_mean_FR += meanFR[j];
        }
        mean_mean_FR /= nodes;
        
        
        /*
         Print fMRI time series
         */
        
        //fprintf(FCout, "%.10f %.10f %.10f %.10f %.10f %.10f %.2f \n\n", G, J_NMDA, w_plus, tmpJi, sigma, global_trans_v, mean_mean_FR);
        for (i=0; i<BOLD_len_i; i++) {
            for (j=0; j<num_output_ts; j++) {
                fprintf(FCout, "%.7f ",BOLD_ex[j][i]);
            }
            fprintf(FCout, "\n");
        }
        fprintf(FCout, "\n");
        fflush(FCout);
        
        
    }
    
    
    _mm_free(n_conn_table);
    _mm_free(region_activity);
    _mm_free(n_conn_table_G_NMDA);
    _mm_free(reg_globinp_p);
    _mm_free(SC_cap);
    
    fclose(FCout);
    //fclose(WFout);
    printf("TVB_C with FIC tuning finished. Execution took %.2f s\n", (float)(time(NULL) - start));
    
    return 0;
}
