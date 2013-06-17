/*
 * This file is part of John the Ripper password cracker,
 * Copyright (c) 2013 by Solar Designer
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted.
 *
 * There's ABSOLUTELY NO WARRANTY, express or implied.
 */

#include <stdio.h> /* for fprintf(stderr, ...) */
#include <stdlib.h>

#include "misc.h" /* for error() */
#include "logger.h"
#include "status.h"
#include "options.h"
#include "rpp.h"
#include "external.h"
#include "cracker.h"
#include "john.h"
#include "mask.h"

static struct mask_context msk_ctx;

void set_mask(struct rpp_context *rpp_ctx, struct db_main *db) {
		
	int i, j, keys_limit, isOptimal = 0x7fffffff;
	int storei[0x100], ctr, flag;
	int comparison_fn_t (const void *a, const void *b) { 
		return ((struct mask_range*)b) -> count - ((struct mask_range*)a) -> count; 
	}
	
	for(i = 0; i < rpp_ctx->count; i++ ) {
		memcpy(msk_ctx.ranges[i].chars, rpp_ctx->ranges[i].chars, 0x100);
		msk_ctx.ranges[i].count = rpp_ctx->ranges[i].count;
		msk_ctx.ranges[i].pos = rpp_ctx->ranges[i].pos - rpp_ctx->output;
		
	}
		
	qsort ((struct mask_range*) msk_ctx.ranges, (size_t)rpp_ctx->count, sizeof(struct mask_range), comparison_fn_t);
	
	keys_limit = 1;
	msk_ctx.count = 0;
	ctr = 0;
	flag = 0;
	for (i = 0; i < rpp_ctx->count; i++) {
		keys_limit *= msk_ctx.ranges[i].count;
		storei[ctr++] = i;
		if (keys_limit > db -> max_int_keys) {
			if (isOptimal > (keys_limit - db -> max_int_keys)) {
				isOptimal = keys_limit - db -> max_int_keys;
				for (j = 0; j < ctr; j++ )
					msk_ctx.activeRangePos[msk_ctx.count + j - flag] = msk_ctx.ranges[storei[j]].pos;
				msk_ctx.count += (ctr - flag);
				flag = 1;
			}	    
			keys_limit /= msk_ctx.ranges[i].count;
			ctr = 0;
			continue;
		}
		else   
		if (isOptimal > (db -> max_int_keys - keys_limit)) {
			for (j = 0; j < ctr; j++ )
				msk_ctx.activeRangePos[msk_ctx.count + j - flag] = msk_ctx.ranges[storei[j]].pos;
				isOptimal = db -> max_int_keys - keys_limit;
				msk_ctx.count += (ctr - flag);
				flag = 0;
				ctr = 0;
			}
	}
	
	db -> msk_ctx = &msk_ctx;
/*	
	for(i = 0; i < msk_ctx.count; i++)
	  printf(" %d ", msk_ctx.activeRangePos[i]);
	
	printf("\n");*/
  
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
	
	if(db -> max_int_keys)
		set_mask(&rpp_ctx, db);
	
	while ((word = msk_next(&rpp_ctx, &msk_ctx))) {
		if (ext_filter(word))
			if (crk_process_key(word))
				break;
	}

	crk_done();

#if 0
	rec_done(event_abort);
#endif
}
