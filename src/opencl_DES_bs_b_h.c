/*
 * This software is Copyright (c) 2012 Sayantan Datta <std2048 at gmail dot com>
 * and it is hereby released to the general public under the following terms:
 * Redistribution and use in source and binary forms, with or without modification, are permitted.
 * Based on Solar Designer implementation of DES_bs_b.c in jtr-v1.7.9
 */


#include <assert.h>
#include <string.h>
#include <sys/time.h>

#include "options.h"
#include "opencl_DES_bs.h"
//#include "mask.h"

#if HARDCODE_SALT

#define LOG_SIZE 1024*16

typedef unsigned WORD vtype;

opencl_DES_bs_transfer *opencl_DES_bs_data;
DES_bs_vector *B;

static cl_kernel krnl[MAX_PLATFORMS * MAX_DEVICES_PER_PLATFORM][4096];
static cl_int err;
static cl_mem index768_gpu, index96_gpu, opencl_DES_bs_data_gpu, B_gpu, cmp_out_gpu, loaded_hash_gpu, transfer_keys_gpu, mask_gpu;
static int set_salt = 0;
static   WORD current_salt;
static size_t DES_local_work_size = WORK_GROUP_SIZE;
static int *loaded_hash;
static unsigned int *cmp_out, num_loaded_hash, min, max ;
static int benchmark = 1;
static unsigned char *input_keys;
static WORD stored_salt[4096]= {0x7fffffff};
static struct mask_context msk_ctx;
static struct db_main *DB;
static unsigned int somethingCracked = 0;
static unsigned int keyCount = 0;

void DES_opencl_clean_all_buffer()
{
	int i;
	const char* errMsg = "Release Memory Object :Failed";
	MEM_FREE(opencl_DES_bs_all);
	MEM_FREE(opencl_DES_bs_data);
	MEM_FREE(B);
	MEM_FREE(loaded_hash);
	MEM_FREE(cmp_out);
	HANDLE_CLERROR(clReleaseMemObject(index768_gpu),errMsg);
	HANDLE_CLERROR(clReleaseMemObject(index96_gpu), errMsg);
	HANDLE_CLERROR(clReleaseMemObject(opencl_DES_bs_data_gpu), errMsg);
	HANDLE_CLERROR(clReleaseMemObject(B_gpu), errMsg);
	HANDLE_CLERROR(clReleaseMemObject(transfer_keys_gpu), errMsg);
	HANDLE_CLERROR(clReleaseMemObject(mask_gpu), errMsg);
	clReleaseMemObject(cmp_out_gpu);
	clReleaseMemObject(loaded_hash_gpu);
	for( i = 0; i < 4096; i++)
		clReleaseKernel(krnl[ocl_gpu_id][i]);

}

void opencl_DES_reset(struct db_main *db) {


	if(db) {
	int i, ctr = 0;
	struct db_salt *salt = db -> salts;

	//if((db->password_count) * 32 > MULTIPLIER) {
	//	fprintf(stderr, "Reduce the number of hashs and try again..\n");
	//	exit(0);
	//}

	do {
			salt -> sequential_id = ctr++;
	} while((salt = salt->next));

	MEM_FREE(loaded_hash);
	MEM_FREE(cmp_out);

	clReleaseMemObject(cmp_out_gpu);
	clReleaseMemObject(loaded_hash_gpu);

	loaded_hash = (int*)mem_alloc((db->password_count)*sizeof(int)*2);
	cmp_out     = (unsigned int*)mem_alloc((db->password_count)*sizeof(unsigned int));

	loaded_hash_gpu = clCreateBuffer(context[ocl_gpu_id], CL_MEM_READ_WRITE, (db->password_count)*sizeof(int)*2, NULL, &err);
	if(loaded_hash_gpu == (cl_mem)0)
		HANDLE_CLERROR(err, "Create Buffer FAILED\n");

	cmp_out_gpu = clCreateBuffer(context[ocl_gpu_id], CL_MEM_READ_WRITE, (db->password_count)*sizeof(unsigned int), NULL, &err);
	if(cmp_out_gpu == (cl_mem)0)
		HANDLE_CLERROR(err, "Create Buffer FAILED\n");

	benchmark = 0;

	/* Expected number of keys to be generated on GPU per work item. Actual number will vary depending on the mask but it should be close */
	db -> max_int_keys = 1000;

	/* Each work item receives one key, so set the following parameters to tuned GWS for format */
	db -> format -> params.max_keys_per_crypt = MULTIPLIER;
	db -> format -> params.min_keys_per_crypt = MULTIPLIER;

	DB = db;

	for (i = 0; i < 4096; i++)
		stored_salt[i] = 0x7fffffff;

	}
}

static void check_mask_descrypt(struct mask_context *msk_ctx) {
	int i, j, k ;
	if(msk_ctx -> count > 8) msk_ctx -> count = 8;

  /* Assumes msk_ctx -> activeRangePos[] is sorted. Check if any range exceeds des key limit */
	for( i = 0; i < msk_ctx->count; i++)
		if(msk_ctx -> activeRangePos[i] >= 8) {
			msk_ctx->count = i;
			break;
		}
	j = 0;
	i = 0;
	k = 0;
 /* Append non-active portion to activeRangePos[] for ease of computation inside GPU */
	while((j <= msk_ctx -> activeRangePos[k]) && (k < msk_ctx -> count)) {
		if(j == msk_ctx -> activeRangePos[k]) {
			k++;
			j++;
			continue;
		}
		msk_ctx -> activeRangePos[msk_ctx -> count + i] = j;
		i++;
		j++;
	}
	while ((i+msk_ctx->count) < 8) {
		msk_ctx -> activeRangePos[msk_ctx -> count + i] = j;
		i++;
		j++;
	}
}

void opencl_DES_bs_init_global_variables() {

	B = (DES_bs_vector*) mem_alloc (MULTIPLIER * 64 * sizeof(DES_bs_vector));
	opencl_DES_bs_all = (opencl_DES_bs_combined*) mem_alloc (MULTIPLIER * sizeof(opencl_DES_bs_combined));
	opencl_DES_bs_data = (opencl_DES_bs_transfer*) mem_alloc (MULTIPLIER * sizeof(opencl_DES_bs_transfer));
	input_keys = (unsigned char *) mem_alloc( MULTIPLIER * 8);
}

void opencl_DES_bs_set_key(char *key, int index)
{
	if(benchmark) {
		unsigned char *dst;
		unsigned int sector,key_index;
		unsigned int flag=key[0];

		sector = index >> DES_BS_LOG2;
		key_index = index & (DES_BS_DEPTH - 1);
		dst = opencl_DES_bs_all[sector].pxkeys[key_index];

		opencl_DES_bs_data[sector].keys_changed = 1;

		dst[0] 				    =	(!flag) ? 0 : key[0];
		dst[sizeof(DES_bs_vector) * 8]	    = 	(!flag) ? 0 : key[1];
		flag = flag&&key[1] ;
		dst[sizeof(DES_bs_vector) * 8 * 2]  =	(!flag) ? 0 : key[2];
		flag = flag&&key[2];
		dst[sizeof(DES_bs_vector) * 8 * 3]  =	(!flag) ? 0 : key[3];
		flag = flag&&key[3];
		dst[sizeof(DES_bs_vector) * 8 * 4]  =	(!flag) ? 0 : key[4];
		flag = flag&&key[4]&&key[5];
		dst[sizeof(DES_bs_vector) * 8 * 5]  =	(!flag) ? 0 : key[5];
		flag = flag&&key[6];
		dst[sizeof(DES_bs_vector) * 8 * 6]  =	(!flag) ? 0 : key[6];
		dst[sizeof(DES_bs_vector) * 8 * 7]  =	(!flag) ? 0 : key[7];
/*
		if (!key[0]) goto fill8;
		*dst = key[0];
		*(dst + sizeof(DES_bs_vector) * 8) = key[1];
		*(dst + sizeof(DES_bs_vector) * 8 * 2) = key[2];
		if (!key[1]) goto fill6;
		if (!key[2]) goto fill5;
		*(dst + sizeof(DES_bs_vector) * 8 * 3) = key[3];
		*(dst + sizeof(DES_bs_vector) * 8 * 4) = key[4];
		if (!key[3]) goto fill4;
		if (!key[4] || !key[5]) goto fill3;
		*(dst + sizeof(DES_bs_vector) * 8 * 5) = key[5];
		if (!key[6]) goto fill2;
		*(dst + sizeof(DES_bs_vector) * 8 * 6) = key[6];
		*(dst + sizeof(DES_bs_vector) * 8 * 7) = key[7];
		return;
fill8:
		dst[0] = 0;
		dst[sizeof(DES_bs_vector) * 8] = 0;
fill6:
		dst[sizeof(DES_bs_vector) * 8 * 2] = 0;
fill5:
		dst[sizeof(DES_bs_vector) * 8 * 3] = 0;
fill4:
		dst[sizeof(DES_bs_vector) * 8 * 4] = 0;
fill3:
		dst[sizeof(DES_bs_vector) * 8 * 5] = 0;
fill2:
		dst[sizeof(DES_bs_vector) * 8 * 6] = 0;
		dst[sizeof(DES_bs_vector) * 8 * 7] = 0;
*/
	}

	else  	{
		keyCount++;
		memcpy(input_keys + 8 * index, key , 8);

	}
}

char *opencl_DES_bs_get_key(int index)
{
	static char out[PLAINTEXT_LENGTH + 1];
	unsigned int sector,block;
	unsigned char *src;
	char *dst;

	if(benchmark) {
	sector = index/DES_BS_DEPTH;
	block  = index%DES_BS_DEPTH;
	init_t();

	src = opencl_DES_bs_all[sector].pxkeys[block];
	dst = out;
	while (dst < &out[PLAINTEXT_LENGTH] && (*dst = *src)) {
		src += sizeof(DES_bs_vector) * 8;
		dst++;
	}
	*dst = 0;
	}

	else {
	  if(index > MULTIPLIER) index = MULTIPLIER - 1;
	 // fprintf(stderr, "Ingetkey:%d\n",index);
	  memcpy(out, input_keys + 8 * index, 8);
	  out[8] = '\0';
	}

	return out;
}


int opencl_DES_bs_cmp_all(WORD *binary, int count)
{
	return 1;
}

inline int opencl_DES_bs_cmp_one(void *binary, int index)
{
	int bit;
	int section = (index >> 5) ;

	if(benchmark) return opencl_DES_bs_cmp_one_b((WORD*)binary, 32, index);
	if(section < min) return 0;
	if(section > max) return 0;

	for(bit = 0; bit < num_loaded_hash; bit++)
		if(cmp_out[bit] == section) return opencl_DES_bs_cmp_one_b((WORD*)binary, 32, index);

	return 0;
}

int opencl_DES_bs_cmp_one_b(WORD *binary, int count, int index)
{
	int bit;
	DES_bs_vector *b;
	int depth;
	unsigned int sector;
	//if(count == 64) printf("cmp exact%d\n",index);
	sector = index >> DES_BS_LOG2;
	index &= (DES_BS_DEPTH - 1);
	depth = index >> 3;
	index &= 7;

	b = (DES_bs_vector *)((unsigned char *)&B[sector * 64] + depth);

#define GET_BIT \
	((unsigned WORD)*(unsigned char *)&b[0] >> index)

	for (bit = 0; bit < 31; bit++, b++)
		if ((GET_BIT ^ (binary[0] >> bit)) & 1)
			return 0;

	for (; bit < count; bit++, b++)
		if ((GET_BIT ^ (binary[bit >> 5] >> (bit & 0x1F))) & 1)
			return 0;

#undef GET_BIT
	return 1;
}

static void find_best_gws(struct fmt_main *fmt)
{
	struct timeval start, end;
	double savetime;
	long int count = 64;
	double speed = 999999, diff;
	int ccount;

	gettimeofday(&start, NULL);
	ccount = count * WORK_GROUP_SIZE * DES_BS_DEPTH;
	opencl_DES_bs_crypt_25(&ccount, NULL);
	gettimeofday(&end, NULL);
	savetime = (end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.000;
	speed = ((double)count) / savetime;
	do {
		count *= 2;
		if ((count * WORK_GROUP_SIZE) > MULTIPLIER) {
			count = count >> 1;
			break;

		}
		gettimeofday(&start, NULL);
		ccount = count * WORK_GROUP_SIZE * DES_BS_DEPTH;
		opencl_DES_bs_crypt_25(&ccount, NULL);
		gettimeofday(&end, NULL);
		savetime = (end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.000;
		diff = (((double)count) / savetime) / speed;
		if (diff < 1) {
			count = count >> 1;
			break;
		}
		diff = diff - 1;
		diff = (diff < 0) ? (-diff) : diff;
		speed = ((double)count) / savetime;
	} while(diff > 0.01);

	if (options.verbosity > 3)
		fprintf(stderr, "Optimal Global Work Size:%ld\n",
		        count * WORK_GROUP_SIZE * DES_BS_DEPTH);

	//fmt -> params.max_keys_per_crypt = DES_BS_DEPTH;
	//fmt -> params.min_keys_per_crypt = DES_BS_DEPTH;
}

	//static char *kernel_source;

	//static int kernel_loaded;

	//static size_t program_size;
/*
static char *include_source(char *pathname, int dev_id, char *options)
{
	static char include[PATH_BUFFER_SIZE];

	sprintf(include, "-I %s %s %s%d %s %s", path_expand(pathname),
	        get_device_type(ocl_gpu_id) == CL_DEVICE_TYPE_CPU ?
	        "-DDEVICE_IS_CPU" : "",
	        "-DDEVICE_INFO=", device_info[ocl_gpu_id],
#ifdef __APPLE__
	        "-DAPPLE",
#else
	        gpu_nvidia(device_info[ocl_gpu_id]) ? "-cl-nv-verbose" : "",
#endif
	        OPENCLBUILDOPTIONS);

	if (options) {
		strcat(include, " ");
		strcat(include, options);
	}

	//fprintf(stderr, "Options used: %s\n", include);
	return include;
}

static void read_kernel_source(char *kernel_filename)
{
	char *kernel_path = path_expand(kernel_filename);
	FILE *fp = fopen(kernel_path, "r");
	size_t source_size, read_size;

	if (!fp)
		fp = fopen(kernel_path, "rb");

	if (!fp)
		HANDLE_CLERROR(!CL_SUCCESS, "Source kernel not found!");

	fseek(fp, 0, SEEK_END);
	source_size = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	MEM_FREE(kernel_source);
	kernel_source = mem_calloc(source_size + 1);
	read_size = fread(kernel_source, sizeof(char), source_size, fp);
	if (read_size != source_size)
		fprintf(stderr,
		    "Error reading source: expected %zu, got %zu bytes.\n",
		    source_size, read_size);
	fclose(fp);
	program_size = source_size;
	kernel_loaded = 1;
}
*/
/*
static void build_kernel_exp(int dev_id, char *options)
{
	//cl_int build_code;
        //char * build_log; size_t log_size;
	const char *srcptr[] = { kernel_source };
	assert(kernel_loaded);
	program[ocl_gpu_id] =
	    clCreateProgramWithSource(context[ocl_gpu_id], 1, srcptr, NULL,
	    &ret_code);

	HANDLE_CLERROR(ret_code, "Error while creating program");

	if(gpu_nvidia(device_info[ocl_gpu_id]))
	   options = "";

	//build_code =
	clBuildProgram(program[ocl_gpu_id], 0, NULL,
		include_source("$JOHN/kernels/", ocl_gpu_id, options), NULL, NULL);
*/
	/*
        HANDLE_CLERROR(clGetProgramBuildInfo(program[ocl_gpu_id], devices[ocl_gpu_id],
                CL_PROGRAM_BUILD_LOG, 0, NULL,
                &log_size), "Error while getting build info I");
        build_log = (char *) mem_alloc((log_size + 1));

	HANDLE_CLERROR(clGetProgramBuildInfo(program[ocl_gpu_id], devices[ocl_gpu_id],
		CL_PROGRAM_BUILD_LOG, log_size + 1, (void *) build_log,
		NULL), "Error while getting build info");

	///Report build errors and warnings
	if (build_code != CL_SUCCESS) {
		//Give us much info about error and exit
		fprintf(stderr, "Compilation log: %s\n", build_log);
		fprintf(stderr, "Error building kernel. Returned build code: %d. DEVICE_INFO=%d\n", build_code, device_info[ocl_gpu_id]);
		HANDLE_CLERROR (build_code, "clBuildProgram failed.");
	}
#ifdef REPORT_OPENCL_WARNINGS
	else if (strlen(build_log) > 1) // Nvidia may return a single '\n' which is not that interesting
		fprintf(stderr, "Compilation log: %s\n", build_log);
#endif
        MEM_FREE(build_log);
#if 0
	FILE *file;
	size_t source_size;
	char *source;

	HANDLE_CLERROR(clGetProgramInfo(program[ocl_gpu_id],
		CL_PROGRAM_BINARY_SIZES,
		sizeof(size_t), &source_size, NULL), "error");
	fprintf(stderr, "source size %zu\n", source_size);
	source = mem_alloc(source_size);

	HANDLE_CLERROR(clGetProgramInfo(program[ocl_gpu_id],
		CL_PROGRAM_BINARIES, sizeof(char *), &source, NULL), "error");

	file = fopen("program.bin", "w");
	if (file == NULL)
		fprintf(stderr, "Error opening binary file\n");
	else if (fwrite(source, source_size, 1, file) != 1)
		fprintf(stderr, "error writing binary\n");
	fclose(file);
	MEM_FREE(source);
#endif
*/
//}

static void init_dev()
{
	char *errMsg = "Create Buffer Failed";
	opencl_init_dev(ocl_gpu_id);

	opencl_DES_bs_data_gpu = clCreateBuffer(context[ocl_gpu_id], CL_MEM_READ_WRITE, MULTIPLIER * sizeof(opencl_DES_bs_transfer), NULL, &err);
	if(opencl_DES_bs_data_gpu == (cl_mem)0)
		HANDLE_CLERROR(err, errMsg);

	index768_gpu = clCreateBuffer(context[ocl_gpu_id], CL_MEM_READ_WRITE, 768 * sizeof(unsigned int), NULL, &err);
	if(index768_gpu == (cl_mem)0)
		HANDLE_CLERROR(err, errMsg);

	index96_gpu = clCreateBuffer(context[ocl_gpu_id], CL_MEM_READ_WRITE, 96 * sizeof(unsigned int), NULL, &err);
	if(index96_gpu == (cl_mem)0)
		HANDLE_CLERROR(err, errMsg);

	B_gpu = clCreateBuffer(context[ocl_gpu_id], CL_MEM_READ_WRITE, 64 * MULTIPLIER * sizeof(DES_bs_vector), NULL, &err);
	if(B_gpu == (cl_mem)0)
		HANDLE_CLERROR(err, errMsg);

	transfer_keys_gpu = clCreateBuffer(context[ocl_gpu_id], CL_MEM_READ_WRITE, 8 * MULTIPLIER , NULL, &err);
	if(transfer_keys_gpu == (cl_mem)0)
		HANDLE_CLERROR(err, errMsg);

	mask_gpu = clCreateBuffer(context[ocl_gpu_id], CL_MEM_READ_WRITE, sizeof(struct mask_context) , NULL, &err);
	if(mask_gpu == (cl_mem)0)
		HANDLE_CLERROR(err, errMsg);

	HANDLE_CLERROR(clEnqueueWriteBuffer(queue[ocl_gpu_id], index768_gpu, CL_TRUE, 0, 768 * sizeof(unsigned int), index768, 0, NULL, NULL ), "Failed Copy data to gpu");

	read_kernel_source("$JOHN/kernels/DES_bs_kernel.cl") ;
}

void modify_src() {

	  int i = 53, j = 1, tmp;
	  static char digits[10] = {'0','1','2','3','4','5','6','7','8','9'} ;
	  static unsigned int  index[48]  = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,
					     24,25,26,27,28,29,30,31,32,33,34,35,
					     48,49,50,51,52,53,54,55,56,57,58,59,
					     72,73,74,75,76,77,78,79,80,81,82,83 } ;
	  for (j = 1; j <= 48; j++) {
		tmp = index96[index[j - 1]] / 10;
		if (tmp == 0)
			kernel_source[i + j * 17] = ' ' ;
		else
			kernel_source[i + j * 17] = digits[tmp];
		tmp = index96[index[j - 1]] % 10;
	     ++i;
	     kernel_source[i + j * 17 ] = digits[tmp];
	     ++i;
	  }
}

void DES_bs_select_device(struct fmt_main *fmt)
{
	init_dev();
	if(!global_work_size)
		find_best_gws(fmt);
	else {
		if (options.verbosity > 3)
			fprintf(stderr, "Global worksize (GWS) forced to %zu\n",
			        global_work_size);
		//fmt -> params.max_keys_per_crypt = global_work_size;
		//fmt -> params.min_keys_per_crypt = WORK_GROUP_SIZE * DES_BS_DEPTH ;
	}
}

void opencl_DES_bs_set_salt(WORD salt)
{
	unsigned int new = salt, section = 0;
	unsigned int old;
	int dst;

	for (section = 0; section < MAX_KEYS_PER_CRYPT / DES_BS_DEPTH; section++) {
	new = salt;
	old = opencl_DES_bs_all[section].salt;
	opencl_DES_bs_all[section].salt = new;
	}
	section = 0;
	current_salt = salt ;
	for (dst = 0; dst < 24; dst++) {
		if ((new ^ old) & 1) {
			DES_bs_vector *sp1, *sp2;
			int src1 = dst;
			int src2 = dst + 24;
			if (new & 1) {
				src1 = src2;
				src2 = dst;
			}
			sp1 = opencl_DES_bs_all[section].Ens[src1];
			sp2 = opencl_DES_bs_all[section].Ens[src2];

			index96[dst] = (WORD *)sp1 - (WORD *)B;
			index96[dst + 24] = (WORD *)sp2 - (WORD *)B;
			index96[dst + 48] = (WORD *)(sp1 + 32) - (WORD *)B;
			index96[dst + 72] = (WORD *)(sp2 + 32) - (WORD *)B;
		}
		new >>= 1;
		old >>= 1;
		if (new == old)
			break;
	}

	set_salt = 1;
}



int opencl_DES_bs_crypt_25(int *pcount, struct db_salt *salt)
{
	int keys_count = *pcount;
	unsigned int section = 0, keys_count_multiple;
	static unsigned int pos, flag = 1 ;
	cl_event evnt;
	size_t N,M;

	if (keys_count%DES_BS_DEPTH == 0)
		keys_count_multiple = keys_count;
	else
		keys_count_multiple = (keys_count / DES_BS_DEPTH + 1) * DES_BS_DEPTH;

	section = keys_count_multiple / DES_BS_DEPTH;

	M = DES_local_work_size;

	if (section % DES_local_work_size != 0)
		N = (section / DES_local_work_size + 1) * DES_local_work_size ;
	//else
		N = MULTIPLIER;

	if (set_salt == 1) {
		unsigned int found = 0;
		if (stored_salt[current_salt] == current_salt) {
			found = 1;
			pos = current_salt;
		}

		if (found == 0) {
			pos = current_salt;
			modify_src();
			clReleaseProgram(program[ocl_gpu_id]);
			//build_kernel( ocl_gpu_id, "-fno-bin-amdil -fno-bin-source -fbin-exe") ;
			opencl_build(ocl_gpu_id, "-cl-opt-disable -fno-bin-amdil -fno-bin-source -fbin-exe", 0, NULL, 1);
			if(benchmark){
				krnl[ocl_gpu_id][pos] = clCreateKernel(program[ocl_gpu_id], "DES_bs_25_bench", &err);
				HANDLE_CLERROR(clSetKernelArg(krnl[ocl_gpu_id][pos], 0, sizeof(cl_mem), &index768_gpu), "Set Kernel Arg FAILED arg0\n");
				HANDLE_CLERROR(clSetKernelArg(krnl[ocl_gpu_id][pos], 1, sizeof(cl_mem), &opencl_DES_bs_data_gpu), "Set Kernel Arg FAILED arg2\n");
				HANDLE_CLERROR(clSetKernelArg(krnl[ocl_gpu_id][pos], 2, sizeof(cl_mem),&B_gpu), "Set Kernel Arg FAILED arg3\n");

			}
			else {
				krnl[ocl_gpu_id][pos] = clCreateKernel(program[ocl_gpu_id], "DES_bs_25", &err);
				HANDLE_CLERROR(clSetKernelArg(krnl[ocl_gpu_id][pos], 0, sizeof(cl_mem), &index768_gpu), "Set Kernel Arg FAILED arg0\n");
				HANDLE_CLERROR(clSetKernelArg(krnl[ocl_gpu_id][pos], 1, sizeof(cl_mem),&B_gpu), "Set Kernel Arg FAILED arg3\n");
				HANDLE_CLERROR(clSetKernelArg(krnl[ocl_gpu_id][pos], 2, sizeof(cl_mem), &loaded_hash_gpu), "Set Kernel krnl Arg 4 :FAILED") ;
				HANDLE_CLERROR(clSetKernelArg(krnl[ocl_gpu_id][pos], 4, sizeof(cl_mem),&cmp_out_gpu), "Set Kernel Arg krnl FAILED arg6\n");
				HANDLE_CLERROR(clSetKernelArg(krnl[ocl_gpu_id][pos], 5, sizeof(cl_mem), &transfer_keys_gpu), "Set Kernel Arg krnl FAILED arg7\n");
				HANDLE_CLERROR(clSetKernelArg(krnl[ocl_gpu_id][pos], 6, sizeof(cl_mem), &mask_gpu), "Set Kernel Arg krnl FAILED arg8\n");
			}
			if (err) {
				fprintf(stderr, "Create Kernel DES_bs_25 FAILED\n");
				return 0;
			}



			stored_salt[current_salt] = current_salt;
		}
		//HANDLE_CLERROR(clEnqueueWriteBuffer(queue[ocl_gpu_id],index96_gpu,CL_TRUE,0,96*sizeof(unsigned int),index96,0,NULL,NULL ), "Failed Copy data to gpu");
		set_salt = 0;

	 printf("NEW SALT\n");
	}

	if(salt) {
		struct db_password *pw;
		int i = 0, *bin;

		pw = salt -> list;
		do {
			  bin = (int *)pw -> binary;
			  loaded_hash[i] = bin[0] ;
			  loaded_hash[i + salt -> count] = bin[1];
			  i++ ;
			  //  printf("%d %d\n", i++, bin[0]);
		} while ((pw = pw -> next)) ;
		num_loaded_hash = (salt -> count);
		//printf("%d\n",loaded_hash[salt->count-1]);
		HANDLE_CLERROR(clEnqueueWriteBuffer(queue[ocl_gpu_id], loaded_hash_gpu, CL_TRUE, 0, (salt -> count) * sizeof(int) * 2, loaded_hash, 0, NULL, NULL ), "Failed Copy data to gpu");
		HANDLE_CLERROR(clEnqueueWriteBuffer(queue[ocl_gpu_id], transfer_keys_gpu, CL_TRUE, 0, 8 * MULTIPLIER, input_keys, 0, NULL, NULL ), "Failed Copy data to gpu");
		HANDLE_CLERROR(clSetKernelArg(krnl[ocl_gpu_id][pos], 3, sizeof(int), &(salt->count)), "Set Kernel krnl Arg 5 :FAILED") ;

		if(flag && !benchmark) {
			if(!DB->msk_ctx) {
				fprintf(stderr, "No given mask.Exiting...\n");
				exit(EXIT_FAILURE);
			}
			memcpy(&msk_ctx, DB->msk_ctx, sizeof(struct mask_context));
			check_mask_descrypt(&msk_ctx);
			for(i = 0; i < 8; i++)
			    printf("%d ",msk_ctx.activeRangePos[i]);
			printf("\n");
			HANDLE_CLERROR(clEnqueueWriteBuffer(queue[ocl_gpu_id], mask_gpu, CL_TRUE, 0, sizeof(struct mask_context), &msk_ctx, 0, NULL, NULL ), "Failed Copy data to gpu");
			flag = 0;
		}

		*pcount = (MULTIPLIER * 32 ) ;

	}

	else {
		int tmp = 0;
		HANDLE_CLERROR(clEnqueueWriteBuffer(queue[ocl_gpu_id],opencl_DES_bs_data_gpu,CL_TRUE,0,MULTIPLIER*sizeof(opencl_DES_bs_transfer),opencl_DES_bs_data,0,NULL,NULL ), "Failed Copy data to gpu");
	}

	err = clEnqueueNDRangeKernel(queue[ocl_gpu_id], krnl[ocl_gpu_id][pos], 1, NULL, &N, &M, 0, NULL, &evnt);
	HANDLE_CLERROR(err, "Enque Kernel Failed");
	clWaitForEvents(1, &evnt);

	if (salt) {
		int i;
		max = ~(unsigned int)0;
		min = MULTIPLIER ;

		HANDLE_CLERROR(clEnqueueReadBuffer(queue[ocl_gpu_id], cmp_out_gpu, CL_TRUE, 0, (salt -> count) * sizeof(unsigned int), cmp_out, 0, NULL, NULL),"Write FAILED\n");
		printf("CMP out %d %d\n", cmp_out[0], salt->sequential_id );
		for (i = 0; i < salt->count ;i++) {
			if(!cmp_out[i]) {
				cmp_out[i] = ~(unsigned int)0;
				continue ;
			}
			cmp_out[i]--;
			if ((int)cmp_out[i] > (int)max)
				max = cmp_out[i];

			if(cmp_out[i] < min)
				min = cmp_out[i];

		}

		if ((int)max>=0) {
			HANDLE_CLERROR(clEnqueueReadBuffer(queue[ocl_gpu_id], B_gpu,CL_TRUE, 0, (salt -> count) * 64 * sizeof(DES_bs_vector), B, 0, NULL, NULL),"Write FAILED\n");
			HANDLE_CLERROR(clEnqueueReadBuffer(queue[ocl_gpu_id], transfer_keys_gpu, CL_TRUE, 0, (salt -> count) * 8 * 32, input_keys, 0, NULL, NULL ), "Failed Copy data from gpu");
			clFinish(queue[ocl_gpu_id]);
			printf("crypt all %d\n",max );
			return (max+1)* DES_BS_DEPTH;
		}

		else return 0;

	}

	else {

		HANDLE_CLERROR(clEnqueueReadBuffer(queue[ocl_gpu_id], B_gpu, CL_TRUE, 0, MULTIPLIER * 64 * sizeof(DES_bs_vector), B, 0, NULL, NULL),"Write FAILED\n");
		clFinish(queue[ocl_gpu_id]);
		return keys_count;
	}
}

#endif
