/*
 * This software is Copyright (c) 2012 Sayantan Datta <std2048 at gmail dot com>
 * and it is hereby released to the general public under the following terms:
 * Redistribution and use in source and binary forms, with or without modification, are permitted.
 * Based on Solar Designer implementation of DES_bs_b.c in jtr-v1.7.9
 */

#include "opencl_DES_common_kenrel.h"

#if !HARDCODE_SALT

#ifndef RV7xx
#define x(p) vxorf(B[ index96[p]], _local_K[_local_index768[p + k] + local_offset_K])
#define y(p, q) vxorf(B[p]       , _local_K[_local_index768[q + k] + local_offset_K])
#else
#define x(p) vxorf(B[index96[p] ], _local_K[index768[p + k] + local_offset_K])
#define y(p, q) vxorf(B[p]       , _local_K[index768[q + k] + local_offset_K])
#endif

#define H1()\
	s1(x(0), x(1), x(2), x(3), x(4), x(5),\
		B,40, 48, 54, 62);\
	s2(x(6), x(7), x(8), x(9), x(10), x(11),\
		B,44, 59, 33, 49);\
	s3(y(7, 12), y(8, 13), y(9, 14),\
		y(10, 15), y(11, 16), y(12, 17),\
		B,55, 47, 61, 37);\
	s4(y(11, 18), y(12, 19), y(13, 20),\
		y(14, 21), y(15, 22), y(16, 23),\
		B,57, 51, 41, 32);\
	s5(x(24), x(25), x(26), x(27), x(28), x(29),\
		B,39, 45, 56, 34);\
	s6(x(30), x(31), x(32), x(33), x(34), x(35),\
		B,35, 60, 42, 50);\
	s7(y(23, 36), y(24, 37), y(25, 38),\
		y(26, 39), y(27, 40), y(28, 41),\
		B,63, 43, 53, 38);\
	s8(y(27, 42), y(28, 43), y(29, 44),\
		y(30, 45), y(31, 46), y(0, 47),\
		B,36, 58, 46, 52);

#define H2()\
	s1(x(48), x(49), x(50), x(51), x(52), x(53),\
		B,8, 16, 22, 30);\
	s2(x(54), x(55), x(56), x(57), x(58), x(59),\
		B,12, 27, 1, 17);\
	s3(y(39, 60), y(40, 61), y(41, 62),\
		y(42, 63), y(43, 64), y(44, 65),\
		B,23, 15, 29, 5);\
	s4(y(43, 66), y(44, 67), y(45, 68),\
		y(46, 69), y(47, 70), y(48, 71),\
		B,25, 19, 9, 0);\
	s5(x(72), x(73), x(74), x(75), x(76), x(77),\
		B,7, 13, 24, 2);\
	s6(x(78), x(79), x(80), x(81), x(82), x(83),\
		B,3, 28, 10, 18);\
	s7(y(55, 84), y(56, 85), y(57, 86),\
		y(58, 87), y(59, 88), y(60, 89),\
		B,31, 11, 21, 6);\
	s8(y(59, 90), y(60, 91), y(61, 92),\
		y(62, 93), y(63, 94), y(32, 95),\
		B,4, 26, 14, 20);

#ifdef _CPU
#define loop_body()\
		H1();\
		if (rounds_and_swapped == 0x100) goto next;\
		H2();\
		k += 96;\
		rounds_and_swapped--;\
		H1();\
		if (rounds_and_swapped == 0x100) goto next;\
		H2();\
		k += 96;\
		rounds_and_swapped--;\
                barrier(CLK_LOCAL_MEM_FENCE);
#elif defined(_NV)
#define loop_body()\
		H1();\
		if (rounds_and_swapped == 0x100) goto next;\
		H2();\
		k += 96;\
		rounds_and_swapped--;\
		barrier(CLK_LOCAL_MEM_FENCE);
#else
#define loop_body()\
		H1();\
		if (rounds_and_swapped == 0x100) goto next;\
		H2();\
		k += 96;\
		rounds_and_swapped--;
#endif

void des_loop(__private vtype *B,
	      __local DES_bs_vector *_local_K,
	      __local ushort *_local_index768,
	      constant uint *index768,
	      __global int *index96,
	      unsigned int iterations,
	      unsigned int local_offset_K) {

		int k, i, rounds_and_swapped;

		for (i = 0; i < 64; i++)
			B[i] = 0;

		k=0;
		rounds_and_swapped = 8;
start:
		loop_body();

		if (rounds_and_swapped > 0) goto start;
		k -= (0x300 + 48);
		rounds_and_swapped = 0x108;
		if (--iterations) goto swap;

		return;

swap:
		H2();
		k += 96;
		if (--rounds_and_swapped) goto start;

next:
		k -= (0x300 - 48);
		rounds_and_swapped = 8;
		iterations--;
		goto start;

}

__kernel void DES_bs_25_self_test( constant uint *index768 __attribute__((max_constant_size(3072))),
			__global int *index96 ,
			__global DES_bs_transfer *DES_bs_all,
			__global DES_bs_vector *B_global)  {

		unsigned int section = get_global_id(0), global_offset_B ,local_offset_K;
		unsigned int local_id = get_local_id(0) ;
		int iterations, i;
		global_offset_B = 64 * section;
		local_offset_K  = 56 * local_id;

		vtype B[64];

		__local DES_bs_vector _local_K[56 * WORK_GROUP_SIZE] ;
#ifndef RV7xx
		__local ushort _local_index768[768] ;
#endif



#ifndef RV7xx
		if (!local_id ) {
			for (i = 0; i < 768; i++)
				_local_index768[i] = index768[i];


		}

		barrier(CLK_LOCAL_MEM_FENCE);
#endif
		DES_bs_finalize_keys_bench(section, DES_bs_all, local_offset_K, _local_K);
		iterations = 25;
		des_loop(B, _local_K, _local_index768, index768, index96, iterations, local_offset_K);
		for (i = 0; i < 64; i++)
			B_global[global_offset_B + i] = (DES_bs_vector)B[i];


}

 __kernel void DES_bs_25_mm( constant uint *index768 __attribute__((max_constant_size(3072))),
			__global int *index96 ,
			__global DES_bs_vector *B_global,
			__global int *binary,
			int num_loaded_hash,
			__global char *transfer_keys,
			__global struct mask_context *msk_ctx,
			__global uint *outKeyIdx)  {

		unsigned int section = get_global_id(0), global_offset_B ,local_offset_K;
		unsigned int local_id = get_local_id(0), activeRangeCount, offset ;
		int iterations, i, loop_count;
		unsigned char input_key[8], activeRangePos[8], rangeNumChars[3], start[3];
		global_offset_B = 64 * section;
		local_offset_K  = 56 * local_id;

		vtype B[64];

		__local DES_bs_vector _local_K[56 * WORK_GROUP_SIZE] ;
#ifndef RV7xx
		__local ushort _local_index768[768] ;
		__local unsigned char range[3*MAX_CHARS];
#endif
		for (i = 0; i < 8 ;i++)
			activeRangePos[i] = msk_ctx[0].activeRangePos[i];

		activeRangeCount = msk_ctx[0].count;

		for (i = 0; i < 8; i++ )
			input_key[i] = transfer_keys[8*section + i];

		for (i = 0; i < activeRangeCount; i++) {
			rangeNumChars[i] = msk_ctx[0].ranges[activeRangePos[i]].count;
			start[i] = msk_ctx[0].ranges[activeRangePos[i]].start;
		}

		loop_count = 1;
		for(i = 0; i < activeRangeCount; i++)
			loop_count *= rangeNumChars[i];

		loop_count = loop_count & 31 ? (loop_count >> 5) + 1: loop_count >> 5;


		if(!section)
			for(i = 0; i < num_loaded_hash; i++)
				outKeyIdx[i] = outKeyIdx[i + num_loaded_hash] = 0;
		barrier(CLK_GLOBAL_MEM_FENCE);

#ifndef RV7xx
		if (!local_id ) {
			for (i = 0; i < 768; i++)
				_local_index768[i] = index768[i];
			for (i = 0; i < MAX_CHARS; i++)
				range[i] = msk_ctx[0].ranges[activeRangePos[0]].chars[i];

			for (i = 0; i < MAX_CHARS; i++)
				range[i + MAX_CHARS] = msk_ctx[0].ranges[activeRangePos[1]].chars[i];

			for (i = 0; i < MAX_CHARS; i++)
				range[i + 2*MAX_CHARS] = msk_ctx[0].ranges[activeRangePos[2]].chars[i];

		}

		barrier(CLK_LOCAL_MEM_FENCE);
#endif

		DES_bs_finalize_keys_passive(local_offset_K, _local_K, activeRangePos, activeRangeCount, input_key);

		offset =0;
		i = 1;

		do {
			DES_bs_finalize_keys_active(local_offset_K, _local_K, offset, activeRangePos, activeRangeCount, range, rangeNumChars, input_key, start);

			iterations = 25;
			des_loop(B, _local_K, _local_index768, index768, index96, iterations, local_offset_K);

			cmp_s( B, binary, num_loaded_hash, B_global, offset, outKeyIdx, section);

			offset = i*32;
			i++;

		} while (i <= loop_count);

}

__kernel void DES_bs_25_om( constant uint *index768 __attribute__((max_constant_size(3072))),
			__global int *index96 ,
			__global DES_bs_transfer *DES_bs_all,
			__global DES_bs_vector *B_global,
			__global int *binary,
			int num_loaded_hash,
			__global uint *outKeyIdx )  {

		unsigned int section = get_global_id(0), global_offset_B ,local_offset_K;
		unsigned int local_id = get_local_id(0) ;
		int iterations, i;
		global_offset_B = 64 * section;
		local_offset_K  = 56 * local_id;

		vtype B[64];

		__local DES_bs_vector _local_K[56 * WORK_GROUP_SIZE] ;
#ifndef RV7xx
		__local ushort _local_index768[768] ;
#endif



#ifndef RV7xx
		if (!local_id ) {
			for (i = 0; i < 768; i++)
				_local_index768[i] = index768[i];


		}

		barrier(CLK_LOCAL_MEM_FENCE);
#endif
		if(!section)
			for(i = 0; i < num_loaded_hash; i++)
				outKeyIdx[i] = outKeyIdx[i + num_loaded_hash] = 0;
		barrier(CLK_GLOBAL_MEM_FENCE);

		DES_bs_finalize_keys_bench(section, DES_bs_all, local_offset_K, _local_K);
		iterations = 25;
		des_loop(B, _local_K, _local_index768, index768, index96, iterations, local_offset_K);
		cmp_s( B, binary, num_loaded_hash, B_global, 0, outKeyIdx, section);

}
#endif
