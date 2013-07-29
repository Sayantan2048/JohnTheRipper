/*
 * This software is Copyright (c) 2013 Sayantan Datta <std2048 at gmail dot com>
 * and it is hereby released to the general public under the following terms:
 * Redistribution and use in source and binary forms, with or without modification, are permitted.
 * This is format is based on mscash-cuda by Lukas Odzioba
 * <lukas dot odzioba at gmail dot com>
 */

#include "opencl_mscash.h"

#define BITMAP_HASH_0 	    (BITMAP_SIZE_0 - 1)
#define BITMAP_HASH_1	    (BITMAP_SIZE_1 - 1)

inline void md4_crypt(__private uint *output, __private uint *nt_buffer)
{
	unsigned int a = INIT_A;
	unsigned int b = INIT_B;
	unsigned int c = INIT_C;
	unsigned int d = INIT_D;

	/* Round 1 */
	a += (d ^ (b & (c ^ d))) + nt_buffer[0];
	a = (a << 3) | (a >> 29);
	d += (c ^ (a & (b ^ c))) + nt_buffer[1];
	d = (d << 7) | (d >> 25);
	c += (b ^ (d & (a ^ b))) + nt_buffer[2];
	c = (c << 11) | (c >> 21);
	b += (a ^ (c & (d ^ a))) + nt_buffer[3];
	b = (b << 19) | (b >> 13);

	a += (d ^ (b & (c ^ d))) + nt_buffer[4];
	a = (a << 3) | (a >> 29);
	d += (c ^ (a & (b ^ c))) + nt_buffer[5];
	d = (d << 7) | (d >> 25);
	c += (b ^ (d & (a ^ b))) + nt_buffer[6];
	c = (c << 11) | (c >> 21);
	b += (a ^ (c & (d ^ a))) + nt_buffer[7];
	b = (b << 19) | (b >> 13);

	a += (d ^ (b & (c ^ d))) + nt_buffer[8];
	a = (a << 3) | (a >> 29);
	d += (c ^ (a & (b ^ c))) + nt_buffer[9];
	d = (d << 7) | (d >> 25);
	c += (b ^ (d & (a ^ b))) + nt_buffer[10];
	c = (c << 11) | (c >> 21);
	b += (a ^ (c & (d ^ a))) + nt_buffer[11];
	b = (b << 19) | (b >> 13);

	a += (d ^ (b & (c ^ d))) + nt_buffer[12];
	a = (a << 3) | (a >> 29);
	d += (c ^ (a & (b ^ c))) + nt_buffer[13];
	d = (d << 7) | (d >> 25);
	c += (b ^ (d & (a ^ b))) + nt_buffer[14];
	c = (c << 11) | (c >> 21);
	b += (a ^ (c & (d ^ a))) + nt_buffer[15];
	b = (b << 19) | (b >> 13);

	/* Round 2 */
	a += ((b & (c | d)) | (c & d)) + nt_buffer[0] + SQRT_2;
	a = (a << 3) | (a >> 29);
	d += ((a & (b | c)) | (b & c)) + nt_buffer[4] + SQRT_2;
	d = (d << 5) | (d >> 27);
	c += ((d & (a | b)) | (a & b)) + nt_buffer[8] + SQRT_2;
	c = (c << 9) | (c >> 23);
	b += ((c & (d | a)) | (d & a)) + nt_buffer[12] + SQRT_2;
	b = (b << 13) | (b >> 19);

	a += ((b & (c | d)) | (c & d)) + nt_buffer[1] + SQRT_2;
	a = (a << 3) | (a >> 29);
	d += ((a & (b | c)) | (b & c)) + nt_buffer[5] + SQRT_2;
	d = (d << 5) | (d >> 27);
	c += ((d & (a | b)) | (a & b)) + nt_buffer[9] + SQRT_2;
	c = (c << 9) | (c >> 23);
	b += ((c & (d | a)) | (d & a)) + nt_buffer[13] + SQRT_2;
	b = (b << 13) | (b >> 19);

	a += ((b & (c | d)) | (c & d)) + nt_buffer[2] + SQRT_2;
	a = (a << 3) | (a >> 29);
	d += ((a & (b | c)) | (b & c)) + nt_buffer[6] + SQRT_2;
	d = (d << 5) | (d >> 27);
	c += ((d & (a | b)) | (a & b)) + nt_buffer[10] + SQRT_2;
	c = (c << 9) | (c >> 23);
	b += ((c & (d | a)) | (d & a)) + nt_buffer[14] + SQRT_2;
	b = (b << 13) | (b >> 19);

	a += ((b & (c | d)) | (c & d)) + nt_buffer[3] + SQRT_2;
	a = (a << 3) | (a >> 29);
	d += ((a & (b | c)) | (b & c)) + nt_buffer[7] + SQRT_2;
	d = (d << 5) | (d >> 27);
	c += ((d & (a | b)) | (a & b)) + nt_buffer[11] + SQRT_2;
	c = (c << 9) | (c >> 23);
	b += ((c & (d | a)) | (d & a)) + nt_buffer[15] + SQRT_2;
	b = (b << 13) | (b >> 19);

	/* Round 3 */
	a += (d ^ c ^ b) + nt_buffer[0] + SQRT_3;
	a = (a << 3) | (a >> 29);
	d += (c ^ b ^ a) + nt_buffer[8] + SQRT_3;
	d = (d << 9) | (d >> 23);
	c += (b ^ a ^ d) + nt_buffer[4] + SQRT_3;
	c = (c << 11) | (c >> 21);
	b += (a ^ d ^ c) + nt_buffer[12] + SQRT_3;
	b = (b << 15) | (b >> 17);

	a += (d ^ c ^ b) + nt_buffer[2] + SQRT_3;
	a = (a << 3) | (a >> 29);
	d += (c ^ b ^ a) + nt_buffer[10] + SQRT_3;
	d = (d << 9) | (d >> 23);
	c += (b ^ a ^ d) + nt_buffer[6] + SQRT_3;
	c = (c << 11) | (c >> 21);
	b += (a ^ d ^ c) + nt_buffer[14] + SQRT_3;
	b = (b << 15) | (b >> 17);

	a += (d ^ c ^ b) + nt_buffer[1] + SQRT_3;
	a = (a << 3) | (a >> 29);
	d += (c ^ b ^ a) + nt_buffer[9] + SQRT_3;
	d = (d << 9) | (d >> 23);
	c += (b ^ a ^ d) + nt_buffer[5] + SQRT_3;
	c = (c << 11) | (c >> 21);
	b += (a ^ d ^ c) + nt_buffer[13] + SQRT_3;
	b = (b << 15) | (b >> 17);

	a += (d ^ c ^ b) + nt_buffer[3] + SQRT_3;
	a = (a << 3) | (a >> 29);
	d += (c ^ b ^ a) + nt_buffer[11] + SQRT_3;
	d = (d << 9) | (d >> 23);
	c += (b ^ a ^ d) + nt_buffer[7] + SQRT_3;
	c = (c << 11) | (c >> 21);
	b += (a ^ d ^ c) + nt_buffer[15] + SQRT_3;
	b = (b << 15) | (b >> 17);

	output[0] = a + INIT_A;
	output[1] = b + INIT_B;
	output[2] = c + INIT_C;
	output[3] = d + INIT_D;
}

inline void prepare_key(__global uint * key, int length, uint * nt_buffer)
{
	int i = 0, nt_index, keychars;
	nt_index = 0;
	for (i = 0; i < (length + 3)/ 4; i++) {
		keychars = key[i];
		nt_buffer[nt_index++] = (keychars & 0xFF) | (((keychars >> 8) & 0xFF) << 16);
		nt_buffer[nt_index++] = ((keychars >> 16) & 0xFF) | ((keychars >> 24) << 16);
	}
	nt_index = length >> 1;
	nt_buffer[nt_index] = (nt_buffer[nt_index] & 0xFF) | (0x80 << ((length & 1) << 4));
	nt_buffer[nt_index + 1] = 0;
	nt_buffer[14] = length << 4;
}

inline void cmp(__global uint *hashes,
	  __global uint *loaded_hashes,
	  __local uint *bitmap0,
	  __local uint *bitmap1,
	  __private uint *hash,
	  __global uint *outKeyIdx,
	  uint gid,
	  uint num_loaded_hashes) {

	uint loaded_hash, i, tmp;

	for(i = 0; i < num_loaded_hashes; i++) {

		loaded_hash = hash[0] & BITMAP_HASH_1;
		tmp = (bitmap0[loaded_hash >> 5] >> (loaded_hash & 31)) & 1U ;
		if(tmp) {

			loaded_hash = hash[1] & BITMAP_HASH_1;
			tmp &= (bitmap1[loaded_hash >> 5] >> (loaded_hash & 31)) & 1U;
			if(tmp) {

				loaded_hash = loaded_hashes[i * 4 + 3];
				if(hash[2] == loaded_hash) {

					loaded_hash = loaded_hashes[i * 4 + 4];
					if(hash[3] == loaded_hash) {

						hashes[i] = hash[0];
						hashes[1 * num_loaded_hashes + i] = hash[1];
						hashes[2 * num_loaded_hashes + i] = hash[2];
						hashes[3 * num_loaded_hashes + i] = hash[3];
						outKeyIdx[i] = gid ;
					}
				}
			}
		}
	}
 }

__kernel void mscash_self_test(__global uint *keys, __global uint *keyIdx, __global uint *salt, __global uint *outBuffer) {

	int gid = get_global_id(0), i;
	int lid = get_local_id(0);
	int numkeys = get_global_size(0);
	uint nt_buffer[16] = { 0 };
	uint output[4] = { 0 };
	uint base = keyIdx[gid];
	uint passwordlength = base & 63;

	keys += base >> 6;

	__local uint login[12];

	if(!lid)
		for(i = 0; i < 12; i++)
			login[i] = salt[i];
	barrier(CLK_LOCAL_MEM_FENCE);

	prepare_key(keys, passwordlength, nt_buffer);
	md4_crypt(output, nt_buffer);
	nt_buffer[0] = output[0];
	nt_buffer[1] = output[1];
	nt_buffer[2] = output[2];
	nt_buffer[3] = output[3];

	for(i = 0; i < 12; i++)
		nt_buffer[i + 4] = login[i];
	md4_crypt(output, nt_buffer);

	outBuffer[gid] = output[0];
	outBuffer[gid + numkeys] = output[1];
	outBuffer[gid + 2 * numkeys] = output[2];
	outBuffer[gid + 3 * numkeys] = output[3];
}

__kernel void mscash(__global uint *keys,
		     __global uint *keyIdx,
		     __global uint *outBuffer,
		     __global uint *outKeyIdx,
		     __global uint *salt,
		     __global uint *loaded_hashes,
		     __global struct bitmap_ctx *bitmap) {

	int gid = get_global_id(0), i;
	int lid = get_local_id(0);
	int numkeys = get_global_size(0);
	uint nt_buffer[16] = { 0 };
	uint output[4] = { 0 };
	uint base = keyIdx[gid];
	uint passwordlength = base & 63;
	uint num_loaded_hashes = loaded_hashes[0];

	keys += base >> 6;

	__local uint login[12];
	__local uint sbitmap0[BITMAP_SIZE_1 >> 5];
	__local uint sbitmap1[BITMAP_SIZE_1 >> 5];

	if(!gid)
		for (i = 0; i < num_loaded_hashes; i++)
			outKeyIdx[i] = outKeyIdx[i + num_loaded_hashes] = 0;

	for(i = 0; i < ((BITMAP_SIZE_1 >> 5) / LWS); i++)
		sbitmap0[i*LWS + lid] = bitmap[0].bitmap0[i*LWS + lid];

	for(i = 0; i < ((BITMAP_SIZE_1 >> 5)/ LWS); i++)
		sbitmap1[i*LWS + lid] = bitmap[0].bitmap1[i*LWS + lid];

	if(!lid)
		for(i = 0; i < 12; i++)
			login[i] = salt[i];
	barrier(CLK_LOCAL_MEM_FENCE);


	prepare_key(keys, passwordlength, nt_buffer);
	md4_crypt(output, nt_buffer);
	nt_buffer[0] = output[0];
	nt_buffer[1] = output[1];
	nt_buffer[2] = output[2];
	nt_buffer[3] = output[3];
	for(i = 0; i < 12; i++)
		nt_buffer[i + 4] = login[i];
	md4_crypt(output, nt_buffer);
	cmp(outBuffer, loaded_hashes, sbitmap0, sbitmap1, output, outKeyIdx, gid, num_loaded_hashes);
}