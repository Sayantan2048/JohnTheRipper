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

/*
 * Mask mode cracker.
 */

#ifndef _JOHN_MASK_H
#define _JOHN_MASK_H

#include "loader.h"

  /* Range of charcters for a placeholder in the mask */
struct mask_range {
  /* Charchters in the range */
	char chars[0x100];
	
  /* Number of charchters in the range */	
	int count;

  /* Postion of the charcters in mask */
	int pos;
};

  /* Simplified mask structure for processing the mask inside a format for password generation */ 
struct mask_context {
  /* Set of mask pacholders selected for processing inside the format */
	struct mask_range ranges[RULE_RANGES_MAX];

  /* Positions in mask for overwriting in the format */ 	  
	int activeRangePos[RULE_RANGES_MAX];
	
  /* Number of postions for overwriting in the format */
	int count;
};

/*
 * Runs the mask mode cracker.
 */
extern void do_mask_crack(struct db_main *db, char *mask);

#endif
