#include "opencl_DES_kernel_params.h"

#define SWAP(a, b) 	\
	tmp = B[a];	\
	B[a] = B[b];	\
	B[b] = tmp;

#define BIG_SWAP() { 	\
	SWAP(0, 32);	\
	SWAP(1, 33);	\
	SWAP(2, 34);	\
	SWAP(3, 35);	\
	SWAP(4, 36);	\
	SWAP(5, 37);	\
	SWAP(6, 38);	\
	SWAP(7, 39);	\
	SWAP(8, 40);	\
	SWAP(9, 41);	\
	SWAP(10, 42);	\
	SWAP(11, 43);	\
	SWAP(12, 44);	\
	SWAP(13, 45);	\
	SWAP(14, 46);	\
	SWAP(15, 47);	\
	SWAP(16, 48);	\
	SWAP(17, 49);	\
	SWAP(18, 50);	\
	SWAP(19, 51);	\
	SWAP(20, 52);	\
	SWAP(21, 53);	\
	SWAP(22, 54);	\
	SWAP(23, 55);	\
	SWAP(24, 56);	\
	SWAP(25, 57);	\
	SWAP(26, 58);	\
	SWAP(27, 59);	\
	SWAP(28, 60);	\
	SWAP(29, 61);	\
	SWAP(30, 62);	\
	SWAP(31, 63);  	\
}

__kernel void DES_bs_final(__global vtype *generated_hashes,
			   __global DES_bs_vector *cracked_hashes,
			   __global int *uncraked_hashes,
			   int num_loaded_hashes,
			   volatile __global uint *hash_ids,
			   volatile __global uint *bitmap)
{
		int i, section = get_global_id(0);
		int global_work_size = get_global_size(0);
		vtype B[64], tmp;

		for (i = 0; i < 64; i++)
			B[i] = generated_hashes[i * global_work_size + section];

		BIG_SWAP();
		cmp(B, uncraked_hashes, num_loaded_hashes, hash_ids, bitmap, cracked_hashes, section);
}
