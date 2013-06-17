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
 * Mask mode cracker.
 */

#ifndef _JOHN_MASK_H
#define _JOHN_MASK_H

#include "loader.h"

struct mask_range {
	char chars[0x100];
	int count;
	int pos;
};

struct mask_context {
	struct mask_range ranges[RULE_RANGES_MAX];
	int activeRangePos[RULE_RANGES_MAX];
	int count;
};

/*
 * Runs the mask mode cracker.
 */
extern void do_mask_crack(struct db_main *db, char *mask);

#endif
