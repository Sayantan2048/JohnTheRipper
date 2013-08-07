/*
 * This software is Copyright (c) 2012 Sayantan Datta <std2048 at gmail dot com>
 * and it is hereby released to the general public under the following terms:
 * Redistribution and use in source and binary forms, with or without modification, are permitted.
 * Based on Solar Designer implementation of DES_bs_b.c in jtr-v1.7.9
 */

#include "opencl_DES_WGS.h"
#include "opencl_device_info.h"
#include "opencl_shared_mask.h"

#define ARCH_WORD     			int
#define DES_BS_DEPTH                    32
#define DES_bs_vector                   ARCH_WORD

#define MAX_CHARS			MAX_GPU_CHARS

typedef unsigned ARCH_WORD vtype ;

#if no_byte_addressable(DEVICE_INFO)
#define RV7xx
#endif

#if gpu_nvidia(DEVICE_INFO)
#define _NV
#endif

#if cpu(DEVICE_INFO)
#define _CPU
#endif

#if 1
#define MAYBE_GLOBAL __global
#else
#define MAYBE_GLOBAL
#endif

typedef struct{

	union {
		unsigned char c[8][8][sizeof(DES_bs_vector)] ;
		DES_bs_vector v[8][8] ;
	} xkeys ;

	int keys_changed ;
} DES_bs_transfer ;

#define vxorf(a, b) 					\
	((a) ^ (b))

#define vnot(dst, a) 					\
	(dst) = ~(a)
#define vand(dst, a, b) 				\
	(dst) = (a) & (b)
#define vor(dst, a, b) 					\
	(dst) = (a) | (b)
#define vandn(dst, a, b) 				\
	(dst) = (a) & ~(b)

#if defined(_NV)||defined(_CPU)
#define vsel(dst, a, b, c) 				\
	(dst) = (((a) & ~(c)) ^ ((b) & (c)))
#else
#define vsel(dst, a, b, c) 				\
	(dst) = bitselect((a),(b),(c))
#endif

#define vshl(dst, src, shift) 				\
	(dst) = (src) << (shift)
#define vshr(dst, src, shift) 				\
	(dst) = (src) >> (shift)

#define vzero 0

#define vones (~(vtype)0)

#define vst(dst, ofs, src) 				\
	*((MAYBE_GLOBAL vtype *)((MAYBE_GLOBAL DES_bs_vector *)&(dst) + (ofs))) = (src)

#define vst_private(dst, ofs, src) 			\
	*((__private vtype *)((__private DES_bs_vector *)&(dst) + (ofs))) = (src)

#define vxor(dst, a, b) 				\
	(dst) = vxorf((a), (b))

#define vshl1(dst, src) 				\
	vshl((dst), (src), 1)

#define kvtype vtype
#define kvand vand
#define kvor vor
#define kvshl1 vshl1
#define kvshl vshl
#define kvshr vshr


#define mask01 0x01010101
#define mask02 0x02020202
#define mask04 0x04040404
#define mask08 0x08080808
#define mask10 0x10101010
#define mask20 0x20202020
#define mask40 0x40404040
#define mask80 0x80808080


#define kvand_shl1_or(dst, src, mask) 			\
	kvand(tmp, src, mask); 				\
	kvshl1(tmp, tmp); 				\
	kvor(dst, dst, tmp)

#define kvand_shl_or(dst, src, mask, shift) 		\
	kvand(tmp, src, mask); 				\
	kvshl(tmp, tmp, shift); 			\
	kvor(dst, dst, tmp)

#define kvand_shl1(dst, src, mask) 			\
	kvand(tmp, src, mask) ;				\
	kvshl1(dst, tmp)

#define kvand_or(dst, src, mask) 			\
	kvand(tmp, src, mask); 				\
	kvor(dst, dst, tmp)

#define kvand_shr_or(dst, src, mask, shift)		\
	kvand(tmp, src, mask); 				\
	kvshr(tmp, tmp, shift); 			\
	kvor(dst, dst, tmp)

#define kvand_shr(dst, src, mask, shift) 		\
	kvand(tmp, src, mask); 				\
	kvshr(dst, tmp, shift)

#define FINALIZE_NEXT_KEY_BIT_0 { 			\
	kvtype m = mask01, va, vb, tmp; 		\
	kvand(va, v0, m); 				\
	kvand_shl1(vb, v1, m); 				\
	kvand_shl_or(va, v2, m, 2); 			\
	kvand_shl_or(vb, v3, m, 3); 			\
	kvand_shl_or(va, v4, m, 4); 			\
	kvand_shl_or(vb, v5, m, 5); 			\
	kvand_shl_or(va, v6, m, 6); 			\
	kvand_shl_or(vb, v7, m, 7); 			\
	kvor(kp[0], va, vb); 				\
}

#define FINALIZE_NEXT_KEY_BIT_1 { 			\
	kvtype m = mask02, va, vb, tmp; 		\
	kvand_shr(va, v0, m, 1); 			\
	kvand(vb, v1, m); 				\
	kvand_shl1_or(va, v2, m); 			\
	kvand_shl_or(vb, v3, m, 2); 			\
	kvand_shl_or(va, v4, m, 3); 			\
	kvand_shl_or(vb, v5, m, 4); 			\
	kvand_shl_or(va, v6, m, 5); 			\
	kvand_shl_or(vb, v7, m, 6); 			\
	kvor(kp[1], va, vb); 				\
}

#define FINALIZE_NEXT_KEY_BIT_2 { 			\
	kvtype m = mask04, va, vb, tmp; 		\
	kvand_shr(va, v0, m, 2); 			\
	kvand_shr(vb, v1, m, 1); 			\
	kvand_or(va, v2, m); 				\
	kvand_shl1_or(vb, v3, m); 			\
	kvand_shl_or(va, v4, m, 2); 			\
	kvand_shl_or(vb, v5, m, 3); 			\
	kvand_shl_or(va, v6, m, 4); 			\
	kvand_shl_or(vb, v7, m, 5); 			\
	kvor(kp[2], va, vb); 				\
}

#define FINALIZE_NEXT_KEY_BIT_3 { 			\
	kvtype m = mask08, va, vb, tmp; 		\
	kvand_shr(va, v0, m, 3); 			\
	kvand_shr(vb, v1, m, 2); 			\
	kvand_shr_or(va, v2, m, 1); 			\
	kvand_or(vb, v3, m); 				\
	kvand_shl1_or(va, v4, m); 			\
	kvand_shl_or(vb, v5, m, 2); 			\
	kvand_shl_or(va, v6, m, 3); 			\
	kvand_shl_or(vb, v7, m, 4); 			\
	kvor(kp[3], va, vb); 				\
}

#define FINALIZE_NEXT_KEY_BIT_4 { 			\
	kvtype m = mask10, va, vb, tmp; 		\
	kvand_shr(va, v0, m, 4); 			\
	kvand_shr(vb, v1, m, 3); 			\
	kvand_shr_or(va, v2, m, 2); 			\
	kvand_shr_or(vb, v3, m, 1); 			\
	kvand_or(va, v4, m); 				\
	kvand_shl1_or(vb, v5, m); 			\
	kvand_shl_or(va, v6, m, 2); 			\
	kvand_shl_or(vb, v7, m, 3); 			\
	kvor(kp[4], va, vb); 				\
}

#define FINALIZE_NEXT_KEY_BIT_5 { 			\
	kvtype m = mask20, va, vb, tmp; 		\
	kvand_shr(va, v0, m, 5); 			\
	kvand_shr(vb, v1, m, 4); 			\
	kvand_shr_or(va, v2, m, 3); 			\
	kvand_shr_or(vb, v3, m, 2); 			\
	kvand_shr_or(va, v4, m, 1); 			\
	kvand_or(vb, v5, m); 				\
	kvand_shl1_or(va, v6, m); 			\
	kvand_shl_or(vb, v7, m, 2); 			\
	kvor(kp[5], va, vb); 				\
}

#define FINALIZE_NEXT_KEY_BIT_6 { 			\
	kvtype m = mask40, va, vb, tmp; 		\
	kvand_shr(va, v0, m, 6); 			\
	kvand_shr(vb, v1, m, 5); 			\
	kvand_shr_or(va, v2, m, 4); 			\
	kvand_shr_or(vb, v3, m, 3); 			\
	kvand_shr_or(va, v4, m, 2); 			\
	kvand_shr_or(vb, v5, m, 1); 			\
	kvand_or(va, v6, m); 				\
	kvand_shl1_or(vb, v7, m); 			\
	kvor(kp[6], va, vb); 				\
}

#define FINALIZE_NEXT_KEY_BIT_7 { 			\
	kvtype m = mask80, va, vb, tmp; 		\
	kvand_shr(va, v0, m, 7); 			\
	kvand_shr(vb, v1, m, 6); 			\
	kvand_shr_or(va, v2, m, 5); 			\
	kvand_shr_or(vb, v3, m, 4); 			\
	kvand_shr_or(va, v4, m, 3); 			\
	kvand_shr_or(vb, v5, m, 2); 			\
	kvand_shr_or(va, v6, m, 1); 			\
	kvand_or(vb, v7, m); 				\
	kvor(kp[7], va, vb); 				\
}

inline void DES_bs_finalize_keys_bench(unsigned int section,
				__global DES_bs_transfer *DES_bs_all,
				int local_offset_K,
				__local DES_bs_vector *K) {

	__local DES_bs_vector *kp = (__local DES_bs_vector *)&K[local_offset_K] ;

	unsigned int ic ;
	kvtype v0, v1, v2, v3, v4, v5, v6, v7;

	for (ic = 0; ic < 8; ic++) {

		MAYBE_GLOBAL DES_bs_vector *vp;
		vp = (MAYBE_GLOBAL DES_bs_vector *)&DES_bs_all[section].xkeys.v[ic][0] ;

		kp = (__local DES_bs_vector *)&K[local_offset_K] + 7 * ic;

		v0 = *(MAYBE_GLOBAL kvtype *)&vp[0];
		v1 = *(MAYBE_GLOBAL kvtype *)&vp[1];
		v2 = *(MAYBE_GLOBAL kvtype *)&vp[2];
		v3 = *(MAYBE_GLOBAL kvtype *)&vp[3];
		v4 = *(MAYBE_GLOBAL kvtype *)&vp[4];
		v5 = *(MAYBE_GLOBAL kvtype *)&vp[5];
		v6 = *(MAYBE_GLOBAL kvtype *)&vp[6];
		v7 = *(MAYBE_GLOBAL kvtype *)&vp[7];

		FINALIZE_NEXT_KEY_BIT_0
		FINALIZE_NEXT_KEY_BIT_1
		FINALIZE_NEXT_KEY_BIT_2
		FINALIZE_NEXT_KEY_BIT_3
		FINALIZE_NEXT_KEY_BIT_4
		FINALIZE_NEXT_KEY_BIT_5
		FINALIZE_NEXT_KEY_BIT_6

	}
}

void load_v_active(__private kvtype *v, unsigned int weight, unsigned int j, unsigned int modulo, __local uchar *range, int idx, uint offset, uint start) {
	unsigned int a, b, c, d;

	a = (j + offset) /  weight ;
	b = (j + 8 + offset) / weight ;
	c = (j + 16 + offset) / weight ;
	d = (j + 24 + offset) / weight ;

	a = a % modulo;
	b = b % modulo;
	c = c % modulo;
	d = d % modulo;

	if(start) {
		a += start;
		b += start;
		c += start;
		d += start;
	}

	else {
		a = range[a + idx*MAX_CHARS];
		b = range[b + idx*MAX_CHARS];
		c = range[c + idx*MAX_CHARS];
		d = range[d + idx*MAX_CHARS];
	}

	v[0] = (a) | (unsigned int)(b << 8) | (unsigned int)(c << 16) | (unsigned int)(d << 24) ;
}

void DES_bs_finalize_keys_active(int local_offset_K,
			   __local DES_bs_vector *K,
			   unsigned int offset,
			   __private uchar *activeRangePos,
			   uint activeRangeCount,
			   __local unsigned char* range,
			   __private uchar *rangeNumChars,
			   __private uchar *input_key,
			   __private uchar *start) {


	__local DES_bs_vector *kp = (__local DES_bs_vector *)&K[local_offset_K] ;

	unsigned int weight, i, ic  ;
	kvtype v0, v1, v2, v3, v4, v5, v6, v7;

	for(ic = 0; ic < activeRangeCount; ic++) {

		kp = (__local DES_bs_vector *)&K[local_offset_K] + 7 * activeRangePos[ic];

		weight = 1;
		i = 0;
		while(i< ic) {
			weight *= rangeNumChars[i];
			i++;
		}

		load_v_active(&v0, weight, 0, rangeNumChars[ic], range, ic, offset, start[ic]);
		load_v_active(&v1, weight, 1, rangeNumChars[ic], range, ic, offset, start[ic]);
		load_v_active(&v2, weight, 2, rangeNumChars[ic], range, ic, offset, start[ic]);
		load_v_active(&v3, weight, 3, rangeNumChars[ic], range, ic, offset, start[ic]);
		load_v_active(&v4, weight, 4, rangeNumChars[ic], range, ic, offset, start[ic]);
		load_v_active(&v5, weight, 5, rangeNumChars[ic], range, ic, offset, start[ic]);
		load_v_active(&v6, weight, 6, rangeNumChars[ic], range, ic, offset, start[ic]);
		load_v_active(&v7, weight, 7, rangeNumChars[ic], range, ic, offset, start[ic]);

		FINALIZE_NEXT_KEY_BIT_0
		FINALIZE_NEXT_KEY_BIT_1
		FINALIZE_NEXT_KEY_BIT_2
		FINALIZE_NEXT_KEY_BIT_3
		FINALIZE_NEXT_KEY_BIT_4
		FINALIZE_NEXT_KEY_BIT_5
		FINALIZE_NEXT_KEY_BIT_6

	}

}

void DES_bs_finalize_keys_passive(int local_offset_K,
			   __local DES_bs_vector *K,
			   __private uchar *activeRangePos,
			   uint activeRangeCount,
			   __private uchar *input_key) {


	__local DES_bs_vector *kp = (__local DES_bs_vector *)&K[local_offset_K] ;

	unsigned int weight, i, ic  ;
	kvtype v0, v1, v2, v3, v4, v5, v6, v7;


	for(ic = activeRangeCount; ic < 8; ic++) {

		kp = (__local DES_bs_vector *)&K[local_offset_K] + 7 * activeRangePos[ic];

		v0 = input_key[activeRangePos[ic]];
		v0 =  (v0) | (unsigned int)(v0 << 8) | (unsigned int)(v0 << 16) | (unsigned int)(v0 << 24) ;
		v1 = v2 = v3 = v4 = v5 = v6 = v7 = v0;

		FINALIZE_NEXT_KEY_BIT_0
		FINALIZE_NEXT_KEY_BIT_1
		FINALIZE_NEXT_KEY_BIT_2
		FINALIZE_NEXT_KEY_BIT_3
		FINALIZE_NEXT_KEY_BIT_4
		FINALIZE_NEXT_KEY_BIT_5
		FINALIZE_NEXT_KEY_BIT_6

	}

}

#if !HARDCODE_SALT

#if defined(_NV) || defined(_CPU)
#include "opencl_sboxes.h"
#else
#include "opencl_sboxes-s.h"
#endif


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

inline void cmp_s( __private vtype *B,
	  __global int *binary,
	  int num_loaded_hash,
	   __global DES_bs_vector *B_global,
	  uint offset,
	  __global uint *outKeyIdx,
	  int section) {


	int value[2] , mask, i, bit;

	for(i = 0 ; i < num_loaded_hash; i++) {

		value[0] = binary[i];
		value[1] = binary[i + num_loaded_hash];

		mask = B[0] ^ -(value[0] & 1);

		for (bit = 1; bit < 32; bit++)
			mask |= B[bit] ^ -((value[0] >> bit) & 1);

		for (; bit < 64; bit += 2) {
			mask |= B[bit] ^ -((value[1] >> (bit & 0x1F)) & 1);
			mask |= B[bit + 1] ^ -((value[1] >> ((bit + 1) & 0x1F)) & 1);

			if (mask == ~(int)0) goto next_hash;
		}

		mask = 64 * i;
		for (bit = 0; bit < 64; bit++)
				B_global[mask + bit] = (DES_bs_vector)B[bit] ;

		outKeyIdx[i] = section | 0x80000000;
		outKeyIdx[i + num_loaded_hash] = offset;

	next_hash: ;
	}

}
#undef GET_BIT

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
