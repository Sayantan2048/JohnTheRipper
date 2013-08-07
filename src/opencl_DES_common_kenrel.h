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

#if defined(_NV) || defined(_CPU)
#include "opencl_sboxes.h"
#else
#include "opencl_sboxes-s.h"
#endif

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
