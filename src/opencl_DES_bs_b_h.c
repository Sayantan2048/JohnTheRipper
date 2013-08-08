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

#if HARDCODE_SALT && !FULL_UNROLL

#define LOG_SIZE 1024*16

typedef unsigned WORD vtype;

/* Common parameters required for all kernels */
static cl_mem index768_gpu, index96_gpu, B_gpu;
DES_bs_vector *B;

/* Variables required for self test */
static cl_mem opencl_DES_bs_data_gpu;
static cl_kernel self_test_kernel[4096];
opencl_DES_bs_transfer *opencl_DES_bs_data;

/* Variables required for cracking kernels */
static cl_mem loaded_hash_gpu, transfer_keys_gpu, buffer_outKeyIdx;
static unsigned int cmp_out = 0, num_loaded_hashes, min, max, keyCount = 0;
static unsigned int *loaded_hashes, *outKeyIdx;
static unsigned char *input_keys;

/* Variables required for mask mode */
static cl_kernel crk_kernel_mm[4096];
static cl_mem mask_gpu;
static struct mask_context msk_ctx;
static struct db_main *DB;

/* Variables required for other modes */
static cl_kernel crk_kernel_om[4096];

/* Other housekeeping variables */
static cl_int err;
static unsigned int keys_changed = 0;
static int set_salt = 0;
static   WORD current_salt;
static size_t DES_local_work_size = WORK_GROUP_SIZE;
static int self_test = 1;
static unsigned int mask_mode = 0;
static WORD stored_salt[4096]= {0x7fffffff};

static int opencl_DES_bs_crypt_25_mm(int *pcount, struct db_salt *salt);
static int opencl_DES_bs_crypt_25_om(int *pcount, struct db_salt *salt);
static char *opencl_DES_bs_get_key_mm(int index);
static char *opencl_DES_bs_get_key_om(int index);
static void opencl_DES_bs_set_key_mm(char *key, int index);

void DES_opencl_clean_all_buffer() {
	int i;
	const char* errMsg = "Release Memory Object :Failed";

	HANDLE_CLERROR(clReleaseMemObject(opencl_DES_bs_data_gpu), errMsg);
	HANDLE_CLERROR(clReleaseMemObject(B_gpu), errMsg);

	if(!self_test) {

		HANDLE_CLERROR(clReleaseMemObject(transfer_keys_gpu), errMsg);
		HANDLE_CLERROR(clReleaseMemObject(mask_gpu), errMsg);
		HANDLE_CLERROR(clReleaseMemObject(loaded_hash_gpu), errMsg);
		HANDLE_CLERROR(clReleaseMemObject(buffer_outKeyIdx), errMsg);
		for( i = 0; i < 4096; i++) {
			if(stored_salt[i] == i) {
				if(mask_mode) HANDLE_CLERROR(clReleaseKernel(crk_kernel_mm[i]), "Release kernel mask_mode failed.");
				else HANDLE_CLERROR(clReleaseKernel(crk_kernel_om[i]), "Release kernel for non-mask modees failed");
			}
		}

		MEM_FREE(loaded_hashes);
		MEM_FREE(outKeyIdx);

	}
	else
		for( i = 0; i < 4096; i++)
			if(stored_salt[i] == i)
				HANDLE_CLERROR(clReleaseKernel(self_test_kernel[i]), "Release kernel self-test failed.");

	HANDLE_CLERROR(clReleaseCommandQueue(queue[ocl_gpu_id]), "Error releasing command queue.");

	MEM_FREE(opencl_DES_bs_all);
	MEM_FREE(opencl_DES_bs_data);
	MEM_FREE(input_keys);
	MEM_FREE(B);
}

void opencl_DES_reset(struct db_main *db) {

	if(db) {
		unsigned int i, ctr = 0, length = 0;
		struct db_salt *salt = db -> salts;

		do {
			salt -> sequential_id = ctr++;
			/*
			 * During cracking the maximum return count for crypt_all is num_loaded_hashes * 32.
			 * However the space is allocated only for MULTIPLIER * 32 number of stuffs.
			 */
			if((salt->count) > MULTIPLIER) {
				fprintf(stderr, "Reduce the number of hashs and try again..\n");
				exit(0);
			}

		} while((salt = salt->next));

		/* Each work item receives one key, so set the following parameters to tuned GWS for format. */
		db -> format -> params.max_keys_per_crypt = MULTIPLIER;
		db -> format -> params.min_keys_per_crypt = MULTIPLIER;

		loaded_hashes = (unsigned int*)mem_alloc((db->password_count) * sizeof(unsigned int) * 2);
		outKeyIdx     = (unsigned int*)mem_calloc((db->password_count) * sizeof(unsigned int) * 2);
		length = ((db->format->params.max_keys_per_crypt) > ((db->password_count) * sizeof(unsigned int) * 2)) ?
			  (db->format->params.max_keys_per_crypt) : ((db->password_count) * sizeof(unsigned int) * 2);

		loaded_hash_gpu = clCreateBuffer(context[ocl_gpu_id], CL_MEM_READ_WRITE, (db->password_count)*sizeof(int)*2, NULL, &ret_code);
		HANDLE_CLERROR(ret_code, "Create Buffer FAILED\n");
		transfer_keys_gpu = clCreateBuffer(context[ocl_gpu_id], CL_MEM_READ_WRITE, 8 * MULTIPLIER , NULL, &ret_code);
		HANDLE_CLERROR(ret_code, "Error creating buffer for input keys.");
		/* buffer_outKeyIdx is multiplexed for use as mask_offset input and keyIdx output */
		buffer_outKeyIdx = clCreateBuffer(context[ocl_gpu_id], CL_MEM_READ_WRITE, length, NULL, &ret_code);
		HANDLE_CLERROR(ret_code, "Error creating buffer compare output\n");
		mask_gpu = clCreateBuffer(context[ocl_gpu_id], CL_MEM_READ_WRITE, sizeof(struct mask_context) , NULL, &ret_code);
		HANDLE_CLERROR(ret_code, "Error creating buffer for mask");

		self_test = 0;

		if(mask_mode) {
			/* Expected number of keys to be generated on GPU per work item. Actual number will vary depending on the mask but it should be close */
			db -> max_int_keys = 1000;

			DB = db;

			db->format->methods.crypt_all = opencl_DES_bs_crypt_25_mm;
			db->format->methods.get_key = opencl_DES_bs_get_key_mm;
			db->format->methods.set_key = opencl_DES_bs_set_key_mm;
		}

		else {
			db->format->methods.crypt_all = opencl_DES_bs_crypt_25_om;
			db->format->methods.set_key = opencl_DES_bs_set_key_self_test;
			db->format->methods.get_key = opencl_DES_bs_get_key_om;

		}

		/* Reset the salts set during self test */
		for( i = 0; i < 4096; i++)
			if(stored_salt[i] == i)
				HANDLE_CLERROR(clReleaseKernel(self_test_kernel[i]), "Release kernel self-test failed.");
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
	/* Append non-active portion to activeRangePos[] for ease of computation inside GPU. */
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

	/* Zeroes the character count for non-active portion. This is helpful in regenerating password in host. */
	for(i = msk_ctx->count; i < 8; i++) {
		msk_ctx->ranges[msk_ctx -> activeRangePos[i]].count = 0;
	}
}

void opencl_DES_bs_init_global_variables() {

	B = (DES_bs_vector*) mem_alloc (MULTIPLIER * 64 * sizeof(DES_bs_vector));
	opencl_DES_bs_all = (opencl_DES_bs_combined*) mem_alloc (MULTIPLIER * sizeof(opencl_DES_bs_combined));
	opencl_DES_bs_data = (opencl_DES_bs_transfer*) mem_alloc (MULTIPLIER * sizeof(opencl_DES_bs_transfer));
	input_keys = (unsigned char *) mem_alloc( MULTIPLIER * 8);
}

/* Used for self test and other modes except mask mode */
void opencl_DES_bs_set_key_self_test(char *key, int index)
{
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

	if(!keys_changed) keys_changed = 1;
}

/* For mask-mode */
static void opencl_DES_bs_set_key_mm(char *key, int index)
{
	keyCount++;
	memcpy(input_keys + 8 * index, key , 8);
	if(!keys_changed) keys_changed = 1;

}

static void passgen(int ctr, int mask_offset, char *key) {
	int i, j, k;

	mask_offset = msk_ctx.flg_wrd ? mask_offset : 0;

	i =  ctr % msk_ctx.ranges[msk_ctx.activeRangePos[0]].count;
	key[msk_ctx.activeRangePos[0] + mask_offset] = msk_ctx.ranges[msk_ctx.activeRangePos[0]].chars[i];

	if (msk_ctx.ranges[msk_ctx.activeRangePos[1]].count) {
		j = (ctr / msk_ctx.ranges[msk_ctx.activeRangePos[0]].count) % msk_ctx.ranges[msk_ctx.activeRangePos[1]].count;
		key[msk_ctx.activeRangePos[1] + mask_offset] = msk_ctx.ranges[msk_ctx.activeRangePos[1]].chars[j];
	}
	if (msk_ctx.ranges[msk_ctx.activeRangePos[2]].count) {
		k = (ctr / (msk_ctx.ranges[msk_ctx.activeRangePos[0]].count * msk_ctx.ranges[msk_ctx.activeRangePos[1]].count)) % msk_ctx.ranges[msk_ctx.activeRangePos[2]].count;
		key[msk_ctx.activeRangePos[2] + mask_offset] = msk_ctx.ranges[msk_ctx.activeRangePos[2]].chars[k];
	}
}

static char *opencl_DES_bs_get_key_mm(int index)
{
	static char out[PLAINTEXT_LENGTH + 1];

	int keyIdx = 0;
	int section = index >> 5;

	if(section < num_loaded_hashes) {
		int section = index >> 5;
		//fprintf(stderr, "InGetKey%0x", index);
		keyIdx = outKeyIdx[section + num_loaded_hashes] + index % 32;
		index = outKeyIdx[section] & 0x7fffffff;
		//fprintf(stderr, "InGetKey%0x %0x ", index, MAX_KEYS_PER_CRYPT);
	}

	index = (index > (MULTIPLIER - 1))? MULTIPLIER - 1 : index;
	memcpy(out, input_keys + 8 * index, 8);

	if(cmp_out && mask_mode)
		passgen(keyIdx, 0, out);

	out[8] = '\0';
	return out;
}

static char *opencl_DES_bs_get_key_om(int index)
{
	static char out[PLAINTEXT_LENGTH + 1];
	unsigned int section,block;
	unsigned char *src;
	char *dst;

	section = index >> 5;

	if(section < num_loaded_hashes)
		index = ((outKeyIdx[section] & 0x7fffffff) << 5) + index % DES_BS_DEPTH;

	index = (index > (MULTIPLIER - 1))? MULTIPLIER - 1 : index;

	section = index/DES_BS_DEPTH;
	block  = index%DES_BS_DEPTH;
	init_t();

	src = opencl_DES_bs_all[section].pxkeys[block];
	dst = out;
	while (dst < &out[PLAINTEXT_LENGTH] && (*dst = *src)) {
		src += sizeof(DES_bs_vector) * 8;
		dst++;
	}
	*dst = 0;
	return out;
}

char *opencl_DES_bs_get_key_self_test(int index)
{
	static char out[PLAINTEXT_LENGTH + 1];
	unsigned int sector,block;
	unsigned char *src;
	char *dst;

	if(self_test) {
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
	int section = (index >> 5) ;

	if(self_test) return opencl_DES_bs_cmp_one_b((WORD*)binary, 32, index);
	if(section < min) return 0;
	if(section > max) return 0;

	if(outKeyIdx[section] > 0) return opencl_DES_bs_cmp_one_b((WORD*)binary, 32, index);

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
	opencl_DES_bs_crypt_25_self_test(&ccount, NULL);
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
		opencl_DES_bs_crypt_25_self_test(&ccount, NULL);
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

static void init_dev()
{
	char *errMsg = "Create Buffer Failed";
	opencl_init_dev(ocl_gpu_id);

	opencl_DES_bs_data_gpu = clCreateBuffer(context[ocl_gpu_id], CL_MEM_READ_WRITE, MULTIPLIER * sizeof(opencl_DES_bs_transfer), NULL, &ret_code);
	HANDLE_CLERROR(ret_code, errMsg);

	index768_gpu = clCreateBuffer(context[ocl_gpu_id], CL_MEM_READ_WRITE, 768 * sizeof(unsigned int), NULL, &ret_code);
	HANDLE_CLERROR(ret_code, errMsg);

	index96_gpu = clCreateBuffer(context[ocl_gpu_id], CL_MEM_READ_WRITE, 96 * sizeof(unsigned int), NULL, &ret_code);
	HANDLE_CLERROR(ret_code, errMsg);

	B_gpu = clCreateBuffer(context[ocl_gpu_id], CL_MEM_READ_WRITE, 64 * MULTIPLIER * sizeof(DES_bs_vector), NULL, &ret_code);
	HANDLE_CLERROR(ret_code, errMsg);

	HANDLE_CLERROR(clEnqueueWriteBuffer(queue[ocl_gpu_id], index768_gpu, CL_TRUE, 0, 768 * sizeof(unsigned int), index768, 0, NULL, NULL ), "Failed Copy data to gpu");

	read_kernel_source("$JOHN/kernels/DES_bs_kernel_1.cl") ;
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

	/* Check if the mask is being used */
	if(options.mask)
		mask_mode = 1;

	if(!global_work_size)
		find_best_gws(fmt);
	else {
		if (options.verbosity > 3)
			fprintf(stderr, "Global worksize (GWS) forced to %zu\n",
			        global_work_size);
		//fmt -> params.max_keys_per_crypt = global_work_size;
		//fmt -> params.min_keys_per_crypt = WORK_GROUP_SIZE * DES_BS_DEPTH ;
	}

	fmt->methods.crypt_all = opencl_DES_bs_crypt_25_self_test;
	fmt->methods.set_key = opencl_DES_bs_set_key_self_test;
	fmt->methods.get_key = opencl_DES_bs_get_key_self_test;

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

int opencl_DES_bs_crypt_25_self_test(int *pcount, struct db_salt *salt)
{
	int keys_count = *pcount;
	unsigned int sections = 0, keys_count_multiple;
	static unsigned int pos;
	cl_event evnt;
	size_t N,M;

	if (keys_count%DES_BS_DEPTH == 0)
		keys_count_multiple = keys_count;
	else
		keys_count_multiple = (keys_count / DES_BS_DEPTH + 1) * DES_BS_DEPTH;

	sections = keys_count_multiple / DES_BS_DEPTH;

	M = DES_local_work_size;

	if (sections % DES_local_work_size != 0)
		N = (sections / DES_local_work_size + 1) * DES_local_work_size ;
	else
		N = sections;

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
			opencl_build(ocl_gpu_id, "-cl-opt-disable -fno-bin-amdil -fno-bin-source -fbin-exe", 0, NULL, 1);
				self_test_kernel[pos] = clCreateKernel(program[ocl_gpu_id], "DES_bs_25_self_test", &err);
				HANDLE_CLERROR(ret_code, "Create Kernel DES_bs_25_self_test FAILED\n");
				/* Self test kernels */
				HANDLE_CLERROR(clSetKernelArg(self_test_kernel[pos], 0, sizeof(cl_mem), &index768_gpu), "Set Kernel Arg FAILED arg0\n");
				HANDLE_CLERROR(clSetKernelArg(self_test_kernel[pos], 1, sizeof(cl_mem), &opencl_DES_bs_data_gpu), "Set Kernel Arg FAILED arg2\n");
				HANDLE_CLERROR(clSetKernelArg(self_test_kernel[pos], 2, sizeof(cl_mem),&B_gpu), "Set Kernel Arg FAILED arg3\n");

			stored_salt[current_salt] = current_salt;
		}
		//HANDLE_CLERROR(clEnqueueWriteBuffer(queue[ocl_gpu_id],index96_gpu,CL_TRUE,0,96*sizeof(unsigned int),index96,0,NULL,NULL ), "Failed Copy data to gpu");
		set_salt = 0;

	 printf("NEW SALT\n");
	}
	if(keys_changed) {
		HANDLE_CLERROR(clEnqueueWriteBuffer(queue[ocl_gpu_id],opencl_DES_bs_data_gpu,CL_TRUE,0,MULTIPLIER*sizeof(opencl_DES_bs_transfer),opencl_DES_bs_data,0,NULL,NULL ), "Failed Copy data to gpu");
		keys_changed = 0;
	}
	HANDLE_CLERROR(clEnqueueNDRangeKernel(queue[ocl_gpu_id], self_test_kernel[pos], 1, NULL, &N, &M, 0, NULL, &evnt), "Enqueue kernel DES_bs_25_self_test failed.\n");
	clWaitForEvents(1, &evnt);

	HANDLE_CLERROR(clEnqueueReadBuffer(queue[ocl_gpu_id], B_gpu, CL_TRUE, 0, MULTIPLIER * 64 * sizeof(DES_bs_vector), B, 0, NULL, NULL),"Write FAILED\n");
	clFinish(queue[ocl_gpu_id]);
	return keys_count;
}

static int opencl_DES_bs_crypt_25_mm(int *pcount, struct db_salt *salt)
{
	int keys_count = *pcount;
	unsigned int sections = 0;
	static unsigned int pos, flag = 1 ;
	struct db_password *pw;
	unsigned int i = 0, *bin;
	cl_event evnt;
	size_t N,M;

	sections = keys_count;
	M = DES_local_work_size;
	if (sections % DES_local_work_size != 0)
		N = (sections / DES_local_work_size + 1) * DES_local_work_size ;
	else
		N = sections;
	N = N > MULTIPLIER ? MULTIPLIER : N;

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
			crk_kernel_mm[pos] = clCreateKernel(program[ocl_gpu_id], "DES_bs_25_mm", &ret_code);
			HANDLE_CLERROR(ret_code, "Create Kernel DES_bs_25_mm FAILED\n");
			HANDLE_CLERROR(clSetKernelArg(crk_kernel_mm[pos], 0, sizeof(cl_mem), &index768_gpu), "Set Kernel Arg FAILED arg0\n");
			HANDLE_CLERROR(clSetKernelArg(crk_kernel_mm[pos], 1, sizeof(cl_mem),&B_gpu), "Set Kernel Arg FAILED arg3\n");
			HANDLE_CLERROR(clSetKernelArg(crk_kernel_mm[pos], 2, sizeof(cl_mem), &loaded_hash_gpu), "Set Kernel krnl Arg 4 :FAILED") ;
			HANDLE_CLERROR(clSetKernelArg(crk_kernel_mm[pos], 4, sizeof(cl_mem), &transfer_keys_gpu), "Set Kernel Arg krnl FAILED arg7\n");
			HANDLE_CLERROR(clSetKernelArg(crk_kernel_mm[pos], 5, sizeof(cl_mem), &mask_gpu), "Set Kernel Arg krnl FAILED arg8\n");
			HANDLE_CLERROR(clSetKernelArg(crk_kernel_mm[pos], 6, sizeof(cl_mem), &buffer_outKeyIdx), "Set Kernel Arg krnl FAILED arg8\n");

			stored_salt[current_salt] = current_salt;
		}

		set_salt = 0;

	 printf("NEW SALT\n");
	}
	pw = salt -> list;
	do {
		  bin = (unsigned int *)pw -> binary;
		  loaded_hashes[i] = bin[0] ;
		  loaded_hashes[i + salt -> count] = bin[1];
		  i++ ;
		  //  printf("%d %d\n", i++, bin[0]);
	} while ((pw = pw -> next)) ;
	num_loaded_hashes = (salt -> count);
	//printf("%d\n",loaded_hashes[salt->count-1]);
	HANDLE_CLERROR(clEnqueueWriteBuffer(queue[ocl_gpu_id], loaded_hash_gpu, CL_TRUE, 0, (salt -> count) * sizeof(int) * 2, loaded_hashes, 0, NULL, NULL ), "Failed Copy data to gpu");
	if(keys_changed){
		HANDLE_CLERROR(clEnqueueWriteBuffer(queue[ocl_gpu_id], transfer_keys_gpu, CL_TRUE, 0, 8 * MULTIPLIER, input_keys, 0, NULL, NULL ), "Failed Copy data to gpu");
		keys_changed = 0;
	}
	HANDLE_CLERROR(clSetKernelArg(crk_kernel_mm[pos], 3, sizeof(int), &(salt->count)), "Set Kernel krnl Arg 5 :FAILED") ;

	if(flag) {
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

	HANDLE_CLERROR(clEnqueueNDRangeKernel(queue[ocl_gpu_id], crk_kernel_mm[pos], 1, NULL, &N, &M, 0, NULL, &evnt), "Enqueue kernel DES_bs_25_mm");
	clWaitForEvents(1, &evnt);

	HANDLE_CLERROR(clEnqueueReadBuffer(queue[ocl_gpu_id], buffer_outKeyIdx, CL_TRUE, 0, (salt->count) * sizeof(unsigned int), outKeyIdx, 0, NULL, NULL), "Write FAILED\n");
	cmp_out = 0;

	// If a positive match is found outKeyIdx contains some positive (gid | 0x80000000) value else contains 0
	for(i = 0; i < ((salt->count) & (~cmp_out)); i++)
		cmp_out = outKeyIdx[i]?0xffffffff:0;

	printf("CMP out %d %d %d\n", cmp_out, (salt->sequential_id), keyCount);

	if (cmp_out) {
		max = 0;
		min = salt->count;
		HANDLE_CLERROR(clEnqueueReadBuffer(queue[ocl_gpu_id], buffer_outKeyIdx, CL_TRUE, 0, 2 * (salt->count) * sizeof(unsigned int), outKeyIdx, 0, NULL, NULL), "Write FAILED\n");
		for (i = 0; i < salt->count ;i++) {
			if (outKeyIdx[i] > 0) {
				max = i;
				if(max < min)
					min = max;
			}
		}
		HANDLE_CLERROR(clEnqueueReadBuffer(queue[ocl_gpu_id], B_gpu, CL_TRUE, 0, (salt -> count) * 64 * sizeof(DES_bs_vector), B, 0, NULL, NULL), "Write FAILED\n");
		clFinish(queue[ocl_gpu_id]);
		printf("crypt all %d\n",max + 1);
		return (max + 1) * DES_BS_DEPTH ;
	}

	else return 0;
}

static int opencl_DES_bs_crypt_25_om(int *pcount, struct db_salt *salt)
{
	int keys_count = *pcount;
	unsigned int sections = 0, keys_count_multiple;
	struct db_password *pw;
	unsigned int i = 0, *bin;
	static unsigned int pos;
	cl_event evnt;
	size_t N,M;

	if (keys_count%DES_BS_DEPTH == 0)
		keys_count_multiple = keys_count;
	else
		keys_count_multiple = (keys_count / DES_BS_DEPTH + 1) * DES_BS_DEPTH;

	sections = keys_count_multiple / DES_BS_DEPTH;

	M = DES_local_work_size;

	if (sections % DES_local_work_size != 0)
		N = (sections / DES_local_work_size + 1) * DES_local_work_size ;
	else
		N = sections;

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
			opencl_build(ocl_gpu_id, "-cl-opt-disable -fno-bin-amdil -fno-bin-source -fbin-exe", 0, NULL, 1);
				crk_kernel_om[pos] = clCreateKernel(program[ocl_gpu_id], "DES_bs_25_om", &ret_code);
				HANDLE_CLERROR(ret_code, "Create Kernel DES_bs_25_om FAILED\n");
				HANDLE_CLERROR(clSetKernelArg(crk_kernel_om[pos], 0, sizeof(cl_mem), &index768_gpu), "Set Kernel Arg FAILED arg0\n");
				HANDLE_CLERROR(clSetKernelArg(crk_kernel_om[pos], 1, sizeof(cl_mem), &opencl_DES_bs_data_gpu), "Set Kernel Arg FAILED arg2\n");
				HANDLE_CLERROR(clSetKernelArg(crk_kernel_om[pos], 2, sizeof(cl_mem),&B_gpu), "Set Kernel Arg FAILED arg3\n");
				HANDLE_CLERROR(clSetKernelArg(crk_kernel_om[pos], 3, sizeof(cl_mem), &loaded_hash_gpu), "Set Kernel krnl Arg 4 :FAILED") ;
				HANDLE_CLERROR(clSetKernelArg(crk_kernel_om[pos], 5, sizeof(cl_mem), &buffer_outKeyIdx), "Set Kernel Arg krnl FAILED arg8\n");

			stored_salt[current_salt] = current_salt;
		}
		set_salt = 0;

	 printf("NEW SALT\n");
	}
	pw = salt -> list;
	do {
		  bin = (unsigned int *)pw -> binary;
		  loaded_hashes[i] = bin[0] ;
		  loaded_hashes[i + salt -> count] = bin[1];
		  i++ ;
		  //  printf("%d %d\n", i++, bin[0]);
	} while ((pw = pw -> next)) ;
	num_loaded_hashes = (salt -> count);
	//printf("%d\n",loaded_hashes[salt->count-1]);
	HANDLE_CLERROR(clEnqueueWriteBuffer(queue[ocl_gpu_id], loaded_hash_gpu, CL_TRUE, 0, (salt -> count) * sizeof(int) * 2, loaded_hashes, 0, NULL, NULL ), "Failed Copy data to gpu");
	HANDLE_CLERROR(clSetKernelArg(crk_kernel_om[pos], 4, sizeof(int), &(salt->count)), "Set Kernel krnl Arg 5 :FAILED") ;
	if (keys_changed) {
		HANDLE_CLERROR(clEnqueueWriteBuffer(queue[ocl_gpu_id],opencl_DES_bs_data_gpu,CL_TRUE,0,MULTIPLIER*sizeof(opencl_DES_bs_transfer),opencl_DES_bs_data,0,NULL,NULL ), "Failed Copy data to gpu");
		keys_changed = 0;
	}
	HANDLE_CLERROR(clEnqueueNDRangeKernel(queue[ocl_gpu_id], crk_kernel_om[pos], 1, NULL, &N, &M, 0, NULL, &evnt), "Enqueue kernel DES_bs_25_om failed.\n");
	clWaitForEvents(1, &evnt);

	HANDLE_CLERROR(clEnqueueReadBuffer(queue[ocl_gpu_id], buffer_outKeyIdx, CL_TRUE, 0, (salt->count) * sizeof(unsigned int), outKeyIdx, 0, NULL, NULL), "Write FAILED\n");
	cmp_out = 0;

	// If a positive match is found outKeyIdx contains some positive (gid | 0x80000000) value else contains 0
	for(i = 0; i < ((salt->count) & (~cmp_out)); i++)
		cmp_out = outKeyIdx[i]?0xffffffff:0;

	printf("CMP out %d %d %d\n", cmp_out, (salt->sequential_id), keyCount);

	if (cmp_out) {
		max = 0;
		min = salt->count;
		HANDLE_CLERROR(clEnqueueReadBuffer(queue[ocl_gpu_id], buffer_outKeyIdx, CL_TRUE, 0, 2 * (salt->count) * sizeof(unsigned int), outKeyIdx, 0, NULL, NULL), "Write FAILED\n");
		for (i = 0; i < salt->count ;i++) {
			if (outKeyIdx[i] > 0) {
				max = i;
				if(max < min)
					min = max;
			}
		}
		HANDLE_CLERROR(clEnqueueReadBuffer(queue[ocl_gpu_id], B_gpu, CL_TRUE, 0, (salt -> count) * 64 * sizeof(DES_bs_vector), B, 0, NULL, NULL), "Write FAILED\n");
		clFinish(queue[ocl_gpu_id]);
		printf("crypt all %d\n",max + 1);
		return (max + 1) * DES_BS_DEPTH ;
	}

	else return 0;
}
#endif
