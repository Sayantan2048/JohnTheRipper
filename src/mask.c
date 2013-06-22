/*
 * This file is part of John the Ripper password cracker,
 * Copyright (c) 2013 by Solar Designer
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted.
 *
 * There's ABSOLUTELY NO WARRANTY, express or implied.
 */

/*
 * This software is Copyright (c) 2013 Sayantan Datta <std2048 at gmail dot com>
 * and it is hereby released to the general public under the following terms:
 * Redistribution and use in source and binary forms, with or without modification, are permitted.
 * There's ABSOLUTELY NO WARRANTY, express or implied.
 */

#include <stdio.h> /* for fprintf(stderr, ...) */
#include <stdlib.h> /* for qsort */

#include "misc.h" /* for error() */
#include "logger.h"
#include "status.h"
#include "options.h"
#include "rpp.h"
#include "external.h"
#include "cracker.h"
#include "john.h"
#include "mask.h"
#include "loader.h"

static struct mask_context msk_ctx;

  /* calculates nCr combinations */ 
void combinationUtil(void *arr, int data[], int start, int end, int index, int r, int target, int *isOptimal);

void calcCombination(void *arr, int n, int target)
{
    int data[n], isOptimal = 0x7fffffff, i;
    ((struct mask_context*)arr) -> count = 0x7fffffff;
    
    for(i = 1; i<= n ;i++) 
		combinationUtil(arr, data, 0, n-1, 0, i, target, &isOptimal);
    
}
 
void combinationUtil(void *arr, int data[], int start, int end, int index, int r, int target, int *isOptimal)
{
    int i;
    
    if (index == r)
    {	int j, tmp = 1;
        for ( j=0; j<r; j++)
             tmp *= ((struct mask_context*)arr) -> ranges[data[j]].count;
	tmp -= target;
	tmp = tmp<0?-tmp:tmp;
	
	if(tmp <= *isOptimal ) {
		if((r < ((struct mask_context*)arr) -> count) || (tmp < *isOptimal)) {
			((struct mask_context*)arr) -> count = r;
			for ( j=0; j<r; j++)
				((struct mask_context*)arr) -> activeRangePos[j] = data[j];
			*isOptimal = tmp;
		}
	}
	return;
    }
    
    for (i=start; i<=end && end-i+1 >= r-index; i++)
    {
        data[index] = ((struct mask_context*)arr) -> ranges[i].pos ;
        combinationUtil(arr, data, i+1, end, index+1, r, target, isOptimal);
    }
} 

static void set_mask(struct rpp_context *rpp_ctx, struct db_main *db) {
		
	int i, j;
	
	for(i = 0; i < rpp_ctx->count; i++ ) {
		memcpy(msk_ctx.ranges[i].chars, rpp_ctx->ranges[i].chars, 0x100);
		msk_ctx.ranges[i].count = rpp_ctx->ranges[i].count;
		msk_ctx.ranges[i].pos = rpp_ctx->ranges[i].pos - rpp_ctx->output;
		
	}
	
	calcCombination(&msk_ctx, rpp_ctx -> count, db -> max_int_keys);
	
	memcpy(db ->msk_ctx, &msk_ctx, sizeof(struct mask_context));
	/*
	for(i = 0; i < msk_ctx.count; i++)
	  printf(" %d ", msk_ctx.activeRangePos[i]);*/
	for(i = 0; i < msk_ctx.count; i++)
			for(j = 0; j < msk_ctx.ranges[msk_ctx.activeRangePos[i]].count; j++)
		printf("%c ",msk_ctx.ranges[msk_ctx.activeRangePos[i]].chars[j]);
	
	printf("\n");
  
}

void do_mask_crack(struct db_main *db, char *mask)
{
	struct rpp_context rpp_ctx;
	char *word;

	if (options.node_count) {
		if (john_main_process)
			fprintf(stderr, "--mask is not yet compatible with --node and --fork\n");
		error();
	}

	log_event("Proceeding with mask mode");

	rpp_init_mask(&rpp_ctx, mask);

	status_init(NULL, 0);
		
#if 0
	rec_restore_mode(restore_state);
	rec_init(db, save_state);

	crk_init(db, fix_state, NULL);
#else
	crk_init(db, NULL, NULL);
#endif
	rpp_process_rule(&rpp_ctx);
	
	db->msk_ctx = (struct mask_context*) mem_alloc(sizeof(struct mask_context));
	
	if(db -> max_int_keys)
		set_mask(&rpp_ctx, db);
	
	while ((word = msk_next(&rpp_ctx, &msk_ctx))) {
		if (ext_filter(word))
			if (crk_process_key(word))
				break;
	}

	crk_done();
	
	MEM_FREE(db -> msk_ctx);

#if 0
	rec_done(event_abort);
#endif
}
