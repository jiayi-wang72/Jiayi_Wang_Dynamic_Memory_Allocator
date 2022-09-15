/**
 * @file mm.c
 * @brief A 64-bit struct-based implicit free list memory allocator
 *  A segregated list of free blocks that help to implement first-fit
 *  in a more efficient manner. A footerless implementation for allocated
 * blocks and mini block for blocks of 16 bytes to improve utilization.
 *
 * 18-213: Introduction to Computer Systems
 *
 * This file uses segregated list to implement dynamic memory allocator,
 * the allocator uses first-fit search method and the free block lists
 * use FIFO insertion policy.
 *
 * An array of global pointer is initialized in mm_init(), which point
 * to the corresponding free_start; mini block (16 bytes) has its own
 * global pointer.
 *
 * The dynamic memory allocator also has footerless allocated block
 * implementation and mini block (16 bytes) to increase utilization.
 * The allocator implements malloc, free, realloc, and calloc allocation-
 * related functions, and the allocator writes to memory to store metadata
 * about the allocation status. The free lists contain blocks whose
 * next + prev fields indicate which block is in the segregated lists.
 *
 * For mini block implementation, the block header contains three status bit,
 * where LSB indicates the current allocation status, the second LSB indicates
 * the previous allocation status, and the third LSB indicates the previous
 * mini block status. In addition to the status bits, the header also contains
 * information about the size of the block so the allocator and calculate
 * and find the next block in the heap.
 *
 *
 * @author Jiayi Wang
 */

#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "memlib.h"
#include "mm.h"

/* Do not change the following! */

#ifdef DRIVER
/* create aliases for driver tests */
#define malloc mm_malloc
#define free mm_free
#define realloc mm_realloc
#define calloc mm_calloc
#define memset mem_memset
#define memcpy mem_memcpy
#endif /* def DRIVER */

/* You can change anything from here onward */

/*
 *****************************************************************************
 * If DEBUG is defined (such as when running mdriver-dbg), these macros      *
 * are enabled. You can use them to print debugging output and to check      *
 * contracts only in debug mode.                                             *
 *                                                                           *
 * Only debugging macros with names beginning "dbg_" are allowed.            *
 * You may not define any other macros having arguments.                     *
 *****************************************************************************
 */
#ifdef DEBUG
/* When DEBUG is defined, these form aliases to useful functions */
#define dbg_printf(...) printf(__VA_ARGS__)
#define dbg_requires(expr) assert(expr)
#define dbg_assert(expr) assert(expr)
#define dbg_ensures(expr) assert(expr)
#define dbg_printheap(...) print_heap(__VA_ARGS__)
#else
/* When DEBUG is not defined, no code gets generated for these */
/* The sizeof() hack is used to avoid "unused variable" warnings */
#define dbg_printf(...) (sizeof(__VA_ARGS__), -1)
#define dbg_requires(expr) (sizeof(expr), 1)
#define dbg_assert(expr) (sizeof(expr), 1)
#define dbg_ensures(expr) (sizeof(expr), 1)
#define dbg_printheap(...) ((void)sizeof(__VA_ARGS__))
#endif

/* Basic constants */

typedef uint64_t word_t;

/** @brief Word and header size (bytes) */
static const size_t wsize = sizeof(word_t);

/** @brief Double word size (bytes) */
static const size_t dsize = 2 * wsize;

/** @brief Minimum block size (bytes) */
static const size_t min_block_size = 2 * dsize;

/**
 * (Must be divisible by dsize)
 * chunksize: heap extend by chunksize-many bytes each time
 */
static const size_t chunksize = (1 << 10);

/**
 * mask to get the bit representing whether the block is free
 */
static const word_t alloc_mask = 0x1;

/**
 * mask to get the bit representing whether the prev block is free
 */
static const word_t prev_mask = 0x2;

/**
 * mask to get the bit representing whether the prev block is mini
 */
static const word_t mini_mask = 0x4;

/**
 * mask to get the bit representing the size of the current block
 */
static const word_t size_mask = ~(word_t)0xF;

/**
 * list index constants
 * from 0 to 13
 * */
static const word_t IDX0 = 0;
static const word_t IDX1 = 1;
static const word_t IDX2 = 2;
static const word_t IDX3 = 3;
static const word_t IDX4 = 4;
static const word_t IDX5 = 5;
static const word_t IDX6 = 6;
static const word_t IDX7 = 7;
static const word_t IDX8 = 8;
static const word_t IDX9 = 9;
static const word_t IDX10 = 10;
static const word_t IDX11 = 11;
static const word_t IDX12 = 12;
static const word_t IDX13 = 13;

/**
 * list index range - upper bound
 * determine the range of a certain segregated list*/
static const word_t RANGE0 = 4;
static const word_t RANGE1 = 6;
static const word_t RANGE2 = 8;
static const word_t RANGE3 = 20;
static const word_t RANGE4 = 32;
static const word_t RANGE5 = 46;
static const word_t RANGE6 = 64;
static const word_t RANGE7 = 66;
static const word_t RANGE8 = 128;
static const word_t RANGE9 = 256;
static const word_t RANGE10 = 512;
static const word_t RANGE11 = 1024;
static const word_t RANGE12 = 2048;

/**
 * the offset of mem_heap_hi() to the pointer to the last block */
static const word_t OFFSET = 7;

/**
 * the number of bucket for regular segregated lists */
static const word_t BUCKET = 14;

typedef struct block block_t;

/** @brief Represents the header and payload of one block in the heap
 * represents the next + prev pointer */
struct block {
    /** @brief Header contains size + allocation flag + prev block
     * allocation flag */
    word_t header;

    union {
        /**
         * @brief A struct of struct block pointer
         * that points to next and prev in normal free block */
        struct {
            block_t *next;
            block_t *prev;
        };
        /**
         * @brief A struct block pointer that points to next
         * in the mini block */
        block_t *next_mini;
        /**
         * @brief A pointer to the block payload.
         */
        char payload[0];
    };
};

/* Global variables */
/** @brief Pointer to first block in the heap */
static block_t *heap_start = NULL;

/** @brief Array of pointer to the block */
static block_t *free_start[BUCKET];

/** @brief Pointer to first block in the mini list */
static block_t *free_mini;

/*
 *****************************************************************************
 * The functions below are short wrapper functions to perform                *
 * bit manipulation, pointer arithmetic, and other helper operations.        *
 *                                                                           *
 * We've given you the function header comments for the functions below      *
 * to help you understand how this baseline code works.                      *
 *                                                                           *
 * Note that these function header comments are short since the functions    *
 * they are describing are short as well; you will need to provide           *
 * adequate details for the functions that you write yourself!               *
 *****************************************************************************
 */

/*
 * ---------------------------------------------------------------------------
 *                        BEGIN SHORT HELPER FUNCTIONS
 * ---------------------------------------------------------------------------
 */

/**
 * @brief Returns the maximum of two integers.
 * @param[in] x
 * @param[in] y
 * @return `x` if `x > y`, and `y` otherwise.
 */
static size_t max(size_t x, size_t y) {
    return (x > y) ? x : y;
}

/**
 * @brief Rounds `size` up to next multiple of n
 * @param[in] size
 * @param[in] n
 * @return The size after rounding up
 */
static size_t round_up(size_t size, size_t n) {
    return n * ((size + (n - 1)) / n);
}

/**
 * @brief Packs the 'size' and 'prev_alloc' and 'alloc' of a block into
 * a word suitable for use as a packed value
 *
 * Packed values are used for both headers and footers.
 *
 * The allocation status is pcked into the lowest bit of the word,
 * the allocation status of previous block is packed into the second
 * lowest bit of the word.
 *
 * @param[in] size The size of the block being presented
 * @param[in] alloc True if the bloc is allocated
 * @param[in] prev_alloc True if the prev bloc is allocated
 * @param[in] mini True if the prev block is mini block
 * @return The packed value
 */
static word_t pack_block(size_t size, bool alloc, bool prev_alloc, bool mini) {
    word_t word = size;
    if (alloc) {
        word |= alloc_mask;
    }
    if (prev_alloc) {
        word |= prev_mask;
    }
    if (mini) {
        word |= mini_mask;
    }
    return word;
}

/**
 * @brief Extracts the size represented in a packed word.
 *
 * This function simply clears the lowest 4 bits of the word, as the heap
 * is 16-byte aligned.
 *
 * @param[in] word A word of size word_t
 * @return The size of the block represented by the word
 */
static size_t extract_size(word_t word) {
    return (word & size_mask);
}

/**
 * @brief Extracts the size of a block from its header.
 * @param[in] block A pointer to the current block
 * @return The size of the block
 */
static size_t get_size(block_t *block) {
    return extract_size(block->header);
}

/**
 * @brief Extracts the prev alloc status of a block from its header
 * @param[in] block A pointer to the current block
 * @return The alloc status of the prev block
 */
static bool get_prev_alloc(block_t *block) {
    word_t header = block->header;
    return header & prev_mask;
}

/**
 * @brief Extracts the prev mini status of a block from its header
 * @param[in] block A pointer to the current block
 * @return the mini status of the prev
 */
static bool get_prev_mini(block_t *block) {
    word_t header = block->header;
    return header & mini_mask;
}
/**
 * @brief Given a payload pointer, returns a pointer to the corresponding
 *        block.
 * @param[in] bp A pointer to a block's payload
 * @return The corresponding block
 */
static block_t *payload_to_header(void *bp) {
    return (block_t *)((char *)bp - offsetof(block_t, payload));
}

/**
 * @brief Given a block pointer, returns a pointer to the corresponding
 *        payload.
 * @param[in] block A pointer to the current block
 * @return A pointer to the block's payload
 * @pre The block must be a valid block, not a boundary tag.
 */
static void *header_to_payload(block_t *block) {
    dbg_requires(get_size(block) != 0);
    return (void *)(block->payload);
}

/**
 * @brief Given a block pointer, returns a pointer to the corresponding
 *        footer.
 * @param[in] block A pointer to the current block
 * @return A pointer to the block's footer
 * @pre The block must be a valid block, not a boundary tag.
 */
static word_t *header_to_footer(block_t *block) {
    dbg_requires(get_size(block) != 0 &&
                 "Called header_to_footer on the epilogue block");
    return (word_t *)(block->payload + get_size(block) - dsize);
}

/**
 * @brief Given a block footer, returns a pointer to the corresponding
 *        header.
 * @param[in] footer A pointer to the block's footer
 * @return A pointer to the start of the block
 * @pre The footer must be the footer of a valid block, not of prologue.
 */
static block_t *footer_to_header(word_t *footer) {
    size_t size = extract_size(*footer);
    dbg_assert(size != 0 && "Called footer_to_header on the prologue block");
    return (block_t *)((char *)footer + wsize - size);
}

/**
 * @brief Returns the payload size of a given block.
 *
 * The payload size is equal to the entire block size minus the sizes of the
 * block's header and footer.
 *
 * @param[in] block A pointer to the current block
 * @return The size of the block's payload
 */
static size_t get_payload_size(block_t *block) {
    size_t asize = get_size(block);
    return asize - wsize;
}

/**
 * @brief Returns the allocation status of a given header value.
 *
 * This is based on the lowest bit of the header value.
 *
 * @param[in] word A word of size word_t
 * @return The allocation status correpsonding to the word
 */
static bool extract_alloc(word_t word) {
    return (bool)(word & alloc_mask);
}

/**
 * @brief Returns the allocation status of a block, based on its header.
 * @param[in] block A pointer to the current block
 * @return The allocation status of the block
 */
static bool get_alloc(block_t *block) {
    return extract_alloc(block->header);
}

/**
 * @brief Writes an epilogue header at the given address.
 *
 * The epilogue header has size 0, and is marked as allocated.
 *
 * @param[in] prev_alloc The alllcation status of prev block
 * @param[in] prev_mini True if the prev block is a mini block
 * @param[out] block The location to write the epilogue header
 * @pre block cannot be NULL pointer
 */
static void write_epilogue(block_t *block, bool prev_alloc, bool prev_mini) {
    dbg_requires(block != NULL);
    dbg_requires((char *)block == mem_heap_hi() - OFFSET);
    block->header = pack_block(0, true, prev_alloc, prev_mini);
}

/**
 * @brief Writes a block starting at the given address.
 *
 * This function writes a header (and footer when the block is free and not
 * mini), where the location of the footer is computed in relation
 * to the header.
 *
 * @param[out] block The location to begin writing the block header
 * @param[in] size The size of the new block
 * @param[in] alloc The allocation status of the new block
 * @param[in] prev_alloc The allocation status of the prev block
 * @param[in] mini_alloc Whether the prev block is a mini block
 * @pre block is not NULL pointer
 * @pre block size should be greater than zero (a valid block)
 */
static void write_block(block_t *block, size_t size, bool alloc,
                        bool prev_alloc, bool mini) {
    dbg_requires(block != NULL);
    dbg_requires(size > 0);
    block->header = pack_block(size, alloc, prev_alloc, mini);
    if (!alloc && (size != dsize)) {
        word_t *footerp = header_to_footer(block);
        *footerp = pack_block(size, alloc, prev_alloc, mini);
    }
}

/**
 * @brief Finds the next consecutive block on the heap.
 *
 * This function accesses the next block in the "implicit list" of the heap
 * by adding the size of the block.
 *
 * @param[in] block A block in the heap
 * @return The next consecutive block on the heap
 * @pre The block is not the epilogue
 */
static block_t *find_next(block_t *block) {
    dbg_requires(block != NULL);
    dbg_requires(get_size(block) != 0 &&
                 "Called find_next on the last block in the heap");
    return (block_t *)((char *)block + get_size(block));
}

/**
 * @brief Find the next consecutive free block in the list
 *
 * This function access the next block in the "explicit free list"
 * by returning the pointer to the next free block.
 *
 * @param[in] block A block in free list
 * @return The next consecutive free block in the free list
 * @pre The block is not NULL
 * */
block_t *find_next_free(block_t *block) {
    dbg_requires(block != NULL);
    if (block->next == block) {
        dbg_printf("block next is block, block is: %ld", (word_t)block);
    }
    return block->next;
}

/**
 * @brief Find the prev consecutive free block in the list
 *
 * This function access the prev block in the "explicit free list"
 * by returning the pointer to the prev free block.
 *
 * @param[in] block A block in free list
 * @return The prev consecutive free block in the free list
 * @pre The block is not NULL
 * */
block_t *find_prev_free(block_t *block) {
    dbg_requires(block != NULL);
    dbg_ensures((block->prev) != block);
    return block->prev;
}

/**
 * @brief Finds the footer of the previous block on the heap.
 * @param[in] block A block in the heap
 * @return The location of the previous block's footer
 */
static word_t *find_prev_footer(block_t *block) {
    // Compute previous footer position as one word before the header
    return &(block->header) - 1;
}

/**
 * @brief Finds the previous consecutive block on the heap.
 *
 * This is the previous block in the "implicit list" of the heap.
 *
 * If the function is called on the first block in the heap, NULL will be
 * returned, since the first block in the heap has no previous block!
 *
 * The position of the previous block is found by reading the previous
 * block's footer to determine its size, then calculating the start of the
 * previous block based on its size.
 *
 * @param[in] block A block in the heap
 * @return The previous consecutive block in the heap.
 */
static block_t *find_prev(block_t *block) {
    dbg_requires(block != NULL);
    word_t *footerp = find_prev_footer(block);
    dbg_printf("prev footer size is: %ld\n", (*footerp | size_mask));

    // Return NULL if called on first block in the heap
    if (extract_size(*footerp) == 0) {
        return NULL;
    }

    block_t *res = footer_to_header(footerp);
    dbg_printf("prev header is: %ld\n", res->header);
    return footer_to_header(footerp);
}

/*
 * ---------------------------------------------------------------------------
 *                        END SHORT HELPER FUNCTIONS
 * ---------------------------------------------------------------------------
 */

/******** The remaining content below are helper and debug routines ********/

/** @brief
 * determine and return the size of the prev block
 * given the pointer to the footer of a block
 * @param[in] block A pointer to the current block
 * @return size of the prev block
 * */
word_t footer_size(block_t *block) {
    word_t *prev_footer = find_prev_footer(block);
    word_t size = extract_size(*prev_footer);
    return size;
}

/** @brief
 * determine whether the footer indicate the
 * block is allocated
 * return true if so and false otherwise
 * @param[in] block A pointer to the current block
 * @return the state whether the prev block is allocated
 * */
bool footer_alloc(block_t *block) {
    word_t *prev_footer = find_prev_footer(block);
    bool alloc = extract_alloc(*prev_footer);
    return alloc;
}

/** @brief
 * insert free block at the start of a list
 * given the start of the list
 * @return
 * @param[in] idx - the index of the global free list pointer
 * in the global array
 * @param[out] block - the pointer to the block being inserted
 * */
void insert_free_block(block_t *block, word_t idx) {
    dbg_requires(block != NULL);

    // if nothing in the list
    if (free_start[idx] == NULL) {
        free_start[idx] = block;
        block->next = block;
        block->prev = block;
    } else { /*list not empty */
        block->next = free_start[idx];
        block->prev = free_start[idx]->prev;
        block->next->prev = block;
        block->prev->next = block;
        free_start[idx] = block;
    }
}

/**
 * @brief insert a free mini block into the free_mini list
 *
 * The mini list is a NULL-ended, sigly linked list,
 * so the function inserts the block at the start of the list
 *
 * @param[out] block A pointer to the inserted block
 * */
void insert_free_mini(block_t *block) {
    // if nothing in the list
    if (free_mini == NULL) {
        block->next_mini = NULL;
        free_mini = block;
    } else { /* list not empty */
        block->next_mini = free_mini;
        free_mini = block;
    }
}

/**
 * @brief
 *
 * determine and return the start of the list
 * given by the size of the free block (including header and footer)
 *
 * @return index of global free list pointer in the global array
 * @param[in] size The size of a block
 * */
word_t get_list_index(word_t size) {
    dbg_requires(min_block_size <= size);
    dbg_requires(size % wsize == 0);

    size = size / wsize;

    if (RANGE0 == size) {
        return IDX0;
    } else if (RANGE1 == size) {
        return IDX1;
    } else if (RANGE2 == size) {
        return IDX2;
    } else if (RANGE2 + 2 <= size && size <= RANGE3) {
        return IDX3;
    } else if (RANGE3 + 2 <= size && size <= RANGE4) {
        return IDX4;
    } else if (RANGE4 + 2 <= size && size <= RANGE5) {
        return IDX5;
    } else if (RANGE5 + 2 <= size && size <= RANGE6) {
        return IDX6;
    } else if (RANGE7 == size) {
        return IDX7;
    } else if (RANGE7 + 2 <= size && size <= RANGE8) {
        return IDX8;
    } else if (RANGE8 + 2 <= size && size <= RANGE9) {
        return IDX9;
    } else if (RANGE9 + 2 <= size && size <= RANGE10) {
        return IDX10;
    } else if (RANGE10 + 2 <= size && size <= RANGE11) {
        return IDX11;
    } else if (RANGE11 + 2 <= size && size <= RANGE12) {
        return IDX12;
    }
    return IDX13;
}

/**
 * @brief
 * This function insert free block at the start of
 * correspoinding segregated free list
 * This function calls helper function
 * 1) insert_free_mini
 * or
 * 2) insert_free_block
 * depending on whether the inserted free block is a mini block
 *
 * @param[out] pointer to the block being inserted
 * */
void insert_free(block_t *block) {
    word_t size = get_size(block);
    if (size == dsize) {
        insert_free_mini(block);
        return;
    }
    word_t idx = get_list_index(size);
    dbg_printf("inserted size is %ld and index is %ld\n", size, idx);
    dbg_printf("inserted location is %ld\n", (word_t)block);
    dbg_printf("inserted block prev: %ld\n", (word_t)block->prev);
    dbg_printf("inserted block next: %ld\n", (word_t)block->next);
    insert_free_block(block, idx);
}

/** @brief
 * This function changes the free_start of a certain segregated free
 * block list, given the index of the pointer in the global array

 * @param[in] index of the free list in the global array
 * @param[in] pointer to the block we would want the free_start of a
 * certain free list points to
 * */
void change_free_start(word_t idx, block_t *block) {
    free_start[idx] = block;
}

/** @brief
 * This function removes a free block from mini block list
 * Since the mini block list is a singly linked list, the function
 * traverses throug the entire free_mini list to find the removed
 * block so the current->next = current->next->next if the next
 * pointer of current block points to found.
 *
 * @param[in] found A pointer to the block being removed
 * @pre free_mini should not be NULL
 * @pre found should not be NULL
 * */
void skip_free_mini(block_t *found) {
    dbg_requires(found != NULL);
    dbg_requires(free_mini != NULL);
    dbg_printf("the skipped block is %ld\n", (word_t)found);
    block_t *block;

    if (free_mini == found) {
        free_mini = found->next_mini;
        dbg_printf("found->next_mini is %ld\n", (word_t)found->next_mini);
        return;
    }

    block_t *prev_block = free_mini;
    for (block = free_mini->next_mini; block != NULL;
         block = block->next_mini) {
        if (block == found) {
            prev_block->next_mini = block->next_mini;
            return;
        }
        prev_block = block;
    }
}

/** @brief
 *
 * This function calls a helper function skip_free_mini if the block
 * being removed is a mini block
 *
 * This function removes a block from a free block list by:
 * prev free block point to next free block
 * next free block point to prev free block
 *
 * @param[in] block A pointer to the block being removed
 * */
void skip_free(block_t *block) {
    dbg_requires(block != NULL);

    word_t size = get_size(block);
    dbg_printf("the size of skipped block is: %ld\n", size);
    if (size == dsize) {
        skip_free_mini(block);
        return;
    }

    word_t idx = get_list_index(get_size(block));
    if (block->next == block) {
        dbg_assert(block->prev = block);
        free_start[idx] = NULL;
    } else {
        block->prev->next = block->next;
        block->next->prev = block->prev;
        free_start[idx] = block->next;
    }
}

/**
 * @brief
 *
 * This function finds the prev mini block
 * by -dsize to the current header address
 *
 * @param[in] block A pointer to the current block
 * @return a pointer to the previous mini block
 * @pre prev block is a mini block
 * */
block_t *find_prev_mini(block_t *block) {
    dbg_requires(block != NULL);
    return (block_t *)((char *)block - dsize);
}

/**
 * @brief
 *
 * This function coalesces the free blocks
 * there are four cases:
 * 1) both allocated
 * 2) prev allocated
 * 3) next allocated
 * 4) both free
 *
 * This function calls other helper functions such as
 * insert_free() and skip_free() to coalesce the consecutive
 * free blocks together
 *
 * This function returns the pointer to the start of the newly
 * coalesced block - either the block passed in or the prev free block
 *
 * The function also writes to prev block, next block, next next block,
 * in different cases depending on whether the information needs to
 * be updated, this is done by write_block and write_epilogue function.
 *
 * @param[out] block
 * @return pointer to the start of the newly coalesced block
 * @pre the block passed in should not be NULL pointer
 */
static block_t *coalesce_block(block_t *block) {
    dbg_printf("we are in the coalesce_block function\n");
    dbg_printf("heap start is: %ld\n", (word_t)heap_start);

    dbg_requires(block != NULL);
    block_t *next_block = find_next(block);
    block_t *prev_block = NULL;
    word_t prev_size;

    bool prev_alloc = get_prev_alloc(block);
    bool prev_mini = get_prev_mini(block);
    bool next_alloc = get_alloc(next_block);

    /* if prev is a free regular block
    it's safe to use find_prev to find the prev block */
    if (!prev_alloc && (!prev_mini)) {
        prev_block = find_prev(block);
        dbg_printf("prev_block is :%ld\n", (word_t)prev_block);
        dbg_assert(prev_block != NULL);
        prev_size = get_size(prev_block);
    }

    /* if prev is a free mini block
    use find_prev_mini to find the prev block */
    if (prev_mini && (!prev_alloc)) {
        prev_size = dsize;
        prev_block = find_prev_mini(block);
        dbg_printf("prev header is: %ld\n", prev_block->header);
        dbg_printf("prev_size is %ld and actual size is %ld\n", prev_size,
                   get_size(prev_block));
        dbg_assert(prev_size = get_size(prev_block));
    }

    word_t block_size = get_size(block);
    dbg_assert(next_block != NULL);
    word_t next_size = get_size(next_block);

    if (!get_alloc(block)) {
        skip_free(block);
        dbg_printf("skipped free and free_mini: %ld\n", (word_t)free_mini);
    }

    // four cases to consider
    if (prev_alloc && next_alloc) {
        // if both prev and next allocated
        dbg_printf("this is the both allocated case\n");
        write_block(block, block_size, false, prev_alloc, prev_mini);
        bool cur_mini = (block_size == dsize);

        // write to next_block to update information
        if (next_size != 0) {
            dbg_printf("cur size is %ld\n", get_size(block));
            write_block(next_block, next_size, next_alloc, false, cur_mini);
        } else {
            write_epilogue(next_block, false, cur_mini);
        }
        insert_free(block);
        return block;

    } else if (prev_alloc) {
        // if prev alloc and next free
        dbg_printf("this is the prev allocated case\n");
        word_t total = block_size + next_size;
        skip_free(next_block);
        block_t *next_next = find_next(next_block);
        bool cur_mini = (total == dsize);
        dbg_assert(!cur_mini);
        write_block(block, total, false, prev_alloc, prev_mini);
        insert_free(block);
        word_t size = get_size(next_next);

        // write to next of next_block to update information
        if (size != 0) {
            write_block(next_next, size, get_alloc(next_next), false, cur_mini);
        } else {
            write_epilogue(next_next, false, cur_mini);
        }
        return block;

    } else if (next_alloc) {
        // if prev free and next alloc
        dbg_printf("this is the next allocated case\n");
        word_t total = block_size + prev_size;
        skip_free(prev_block);
        bool prev_free = get_prev_alloc(prev_block);
        bool prev_prev_mini = get_prev_mini(prev_block);
        bool cur_mini = (total == dsize);
        dbg_assert(!cur_mini);
        write_block(prev_block, total, false, prev_free, prev_prev_mini);

        // write to next_block to update information
        if (next_size != 0) {
            write_block(next_block, next_size, next_alloc, false, cur_mini);
        } else {
            write_epilogue(next_block, false, cur_mini);
        }
        insert_free(prev_block);
        return prev_block;

    } else {
        // if both prev and next are alloc
        dbg_printf("this is the both free case\n");
        word_t total = block_size + prev_size + next_size;
        skip_free(prev_block);
        bool prev_free = get_prev_alloc(prev_block);
        bool prev_prev_mini = get_prev_mini(prev_block);
        skip_free(next_block);
        write_block(prev_block, total, false, prev_free, prev_prev_mini);
        insert_free(prev_block);

        block_t *next_next = find_next(next_block);
        bool cur_mini = (total == dsize);
        dbg_assert(!cur_mini);
        word_t size = get_size(next_next);

        // write to next of next_block to update information
        if (size != 0) {
            write_block(next_next, size, get_alloc(next_next), false, cur_mini);
        } else {
            write_epilogue(next_next, false, cur_mini);
        }
        return prev_block;
    }

    dbg_ensures(mm_checkheap(__LINE__));
    return block;
}

/**
 * @brief
 *
 * Taking in the size to be extended, this function extend the
 * heap space, writes new epilogue, and coalesce the newly allocated
 * space with previous consecutive free block (if necessary)
 *
 * In the function, we write to the header (and footer), insert the
 * free block into the free lists
 *
 * @param[in] size The size of extend_heap request
 * @return the pointer to the start of the extended heap
 */
static block_t *extend_heap(size_t size) {
    void *bp;

    // Allocate an even number of words to maintain alignment
    size = round_up(size, dsize);
    if ((bp = mem_sbrk(size)) == (void *)-1) {
        return NULL;
    }

    // Initialize free block header/footer
    block_t *block = payload_to_header(bp);

    bool prev_alloc = get_prev_alloc(block);
    bool prev_mini = get_prev_mini(block);
    if (prev_alloc) {
        dbg_printf("when calling extend_heap, prev_alloc is true\n");
    }

    // write to the newly allocated header
    if ((word_t)block == (word_t)heap_start) {
        // initialize prev_alloc as true and prev_mini as false
        dbg_printf("should fall into this category\n");
        write_block(block, size, false, true, false);
    } else { /*not the heap start */
        dbg_printf("%ld\n", (word_t)block);
        bool prev_mini = get_prev_mini(block);
        write_block(block, size, false, prev_alloc, prev_mini);
    }
    dbg_printf("this is the block header: %ld\n", (word_t)block->header);
    insert_free(block);

    dbg_printf("extend_heap write to block\n");

    // Create new epilogue header
    block_t *block_next = find_next(block);
    write_epilogue(block_next, prev_alloc, prev_mini);

    // Coalesce in case the previous block was free
    dbg_printf("just allocated new heap space\n");
    block = coalesce_block(block);

    return block;
}

/**
 * @brief
 *
 * This function is called when there might be need to split the block
 * if the actual heap space needed for the block is less than that the
 * block actually contains (at least min_block_size lower), the function
 * would write part of the block as free and insert the new free block into
 * the free list
 *
 * @param[out] block A pointer to the allocated block
 * @param[in] asize the size of the actually used block space
 * @pre block should not be NULL
 * @pre the block being passed in should be allocated
 */
static void split_block(block_t *block, size_t asize) {
    dbg_requires(block != NULL);
    dbg_requires(get_alloc(block));

    size_t block_size = get_size(block);

    if ((block_size - asize) >= dsize) {
        /*if it is possible to split the block
        split the block and case on new free block size
        to determine whether writes new free as free list or free_mini */
        dbg_printf("we can further split the block\n");
        block_t *block_next;

        skip_free(block);
        bool prev_alloc = get_prev_alloc(block);
        bool prev_mini = get_prev_mini(block);
        write_block(block, asize, true, prev_alloc, prev_mini);

        block_next = find_next(block);
        bool cur_mini = (asize == dsize);
        write_block(block_next, block_size - asize, false, true, cur_mini);

        block_t *next_next = find_next(block_next);
        bool next_alloc = get_alloc(next_next);
        word_t next_size = get_size(next_next);
        bool next_mini = ((block_size - asize) == dsize);
        if (next_size != 0) {
            write_block(next_next, next_size, next_alloc, false, next_mini);
        } else {
            write_epilogue(next_next, false, next_mini);
        }

        insert_free(block_next);
        dbg_ensures(!get_alloc(block_next));
        dbg_ensures(mm_checkheap(__LINE__));
    } else { /*not possible to split */
        skip_free(block);
    }

    dbg_ensures(get_alloc(block));
}

/**
 * @brief
 *
 * This function finds a fit for the new allocation of block
 * The function uses first-fit method to find the free block
 * In segregated list, the function starts by looking at the best-fit-sized
 * free list and return block if there is one
 * Otherwise, the function continues to search in a bigger free list
 * until it finds one
 *
 * If the size is the size of the mini block, return the first
 * available free mini block in the free_mini list
 *
 * The function returns NULL if there is no block found
 *
 * @param[in] asize The size needed for the allocated block
 * @return block A pointer to the block being allocated
 */
static block_t *find_fit(size_t asize) {
    block_t *block;

    // if asize is 16, look into mini_free list first
    if (asize == dsize) {
        if (free_mini != NULL) {
            return free_mini;
        }
        asize = min_block_size;
    }

    // get index in the free lists
    word_t cur_start_idx = get_list_index(asize);

    // continue search while not reaching the last bucket
    while (cur_start_idx < BUCKET) {
        block_t *cur_start = free_start[cur_start_idx];
        if (cur_start != NULL) {

            /* start from cur_start->prev (last free block)
            to search for a fit and return when find one */
            for (block = cur_start->prev; block != cur_start;
                 block = block->prev) {

                dbg_assert(block != NULL);
                dbg_assert(!(get_alloc(block)));
                if (!(get_alloc(block)) && (asize <= get_size(block))) {
                    return block;
                }
            }
            if (!(get_alloc(cur_start)) && (asize <= get_size(cur_start))) {
                return cur_start;
            }
        }
        // find the next linked list
        cur_start_idx++;
    }

    return NULL; // no fit found
}

/**
 * @brief
 *
 * check if the prologue contain correct information
 * check if alloc is true
 * chech if size is 0
 * @return false if prologue is wrong or true otherwise
 * */

bool check_prologue() {
    word_t *prologue = find_prev_footer(heap_start);
    if (extract_size(*prologue) != 0) {
        return false;
    }
    if (extract_alloc(*prologue) == false) {
        return false;
    }
    return true;
}

/**
 * @brief
 *
 * check if the epilogue contain correct information
 * check if alloc is true
 * chech if size is 0
 * @return false if epilogue is wrong or true otherwise
 * */
bool check_epilogue() {
    char *epi_address = mem_heap_hi() - OFFSET;
    word_t *epilogue = (word_t *)epi_address;
    if (extract_size(*epilogue) != 0) {
        return false;
    }
    if (extract_alloc(*epilogue) == false) {
        return false;
    }
    return true;
}

/**
 * @brief
 * This helper function checks whether the heap is correct and
 * prints out an error message if the heap has error
 *
 * things that are being checked:
 * address allignment
 * address within heap boundary
 * header + footer consistency
 * no consecutive free blocks
 *
 * @param[in] line The line number mm_checkheap is called on
 * @return true if the heap is correct and false otherwise
 *
 * */
bool heap_check(int line) {
    bool result = true;
    block_t *block;
    for (block = heap_start; get_size(block) > 0; block = find_next(block)) {
        // check allignment
        void *cur = header_to_payload(block);
        if ((word_t)cur % dsize != 0) {
            dbg_printf("line %d: payload is not alligned\n", line);
            result = false;
        }

        // check whether block is within the heap bound
        word_t *footerp = header_to_footer(block);
        if ((word_t)footerp > (word_t)mem_heap_hi()) {
            dbg_printf("line %d: footer is outside the heap\n", line);
            result = false;
        }
        if ((word_t)block < (word_t)mem_heap_lo()) {
            dbg_printf("line %d: header is outside the heap\n", line);
            result = false;
        }

        // check if the size is at least min_block_size large
        word_t size = get_size(block);
        if (size < dsize) {
            dbg_printf("line %d: has size less than dsize\n", line);
            result = false;
        }

        // check if header and footer (size) consistent
        if (!get_alloc(block) && (get_size(block) != dsize)) {
            word_t header = *((word_t *)block);
            word_t footer = *(header_to_footer(block));
            if (extract_size(header) != extract_size(footer)) {
                dbg_printf("line %d: header and footer size do not match\n",
                           line);
                result = false;
            }

            // check if header and footer (alloc) consistent
            if (extract_alloc(header) != extract_alloc(footer)) {
                dbg_printf("line %d: header and footer alloc do not mathc\n",
                           line);
                result = false;
            }

            // check there is no consecutive free block in the heap
            word_t *next = (word_t *)find_next(block);
            if (extract_alloc(header) == false) {
                if (extract_alloc(*next) == false) {
                    dbg_printf("line %d: consecutive blocks are free\n", line);
                    dbg_printf("current pointer: %ld\n", (word_t)block);
                    dbg_printf("current alloc: %ld\n", header & 2);
                    dbg_printf("next pointer: %ld\n", (word_t)next);
                    dbg_printf("next alloc: %ld\n", (*next) & 2);
                    result = false;
                }
            }
        }
    }
    return result;
}

/**
 * @brief
 *
 * This function checks if the block is a legal block
 * in the free segregated list
 * Checks the following:
 * next/prev consistency
 * pointer within the heap memory
 *
 * @param[in] line Line number of the call to mm_checkheap
 * @param[in] block A pointer to the current block
 * @return True if block is legal and false otherwise
 * */
bool check_block(block_t *block, int line) {
    dbg_requires(block != NULL);

    bool result = true;

    //  checking next pointer
    if (block->next != NULL) {
        if (block->next->prev != block) {
            result = false;
            dbg_printf("line %d: block next/prev inconsistent", line);
        }

        if ((word_t)block->next < (word_t)mem_heap_lo() ||
            (word_t)block->next > (word_t)mem_heap_hi()) {
            result = false;
            dbg_printf("line %d: block pointer outside the heap", line);
        }
    }

    // checking prev pointer
    if (block->prev != NULL) {
        if (block->prev->next != block) {
            result = false;
            dbg_printf("line %d: block next/prev inconsistent", line);
        }

        if ((word_t)block->prev < (word_t)mem_heap_lo() ||
            (word_t)block->prev > (word_t)mem_heap_hi()) {
            result = false;
            dbg_printf("line %d: block pointer outside the heap", line);
        }
    }
    return result;
}

/**
 * @brief
 *
 * When called in dbg mode, this function checks whether the current
 * heap has error, it also prints out specific error message.
 *
 * This function checks:
 * 1) helper function check_epilogue and check_prologue
 * epilogue and prologue
 * 2) helper function heap_check
 * block's address alignment
 * whether block lie within the heap boundaries
 * footer and header consistency - size + alloc
 * consecutive free block
 * 3) helper function check_block
 * prev/next consistency
 * free list pointer within heap boundaries
 * 4) check within mm_checkheap
 * free block in heap match free block in lists
 * free block in the correct segregated list
 *
 * @param[in] line The line number mm_checkheap is called
 * @return true if no error and false other wise
 */
bool mm_checkheap(int line) {
    bool result = true;

    block_t *block;
    // check if there is correct prologue
    if (!check_prologue()) {
        dbg_printf("line %d: prologue has error\n", line);
        result = false;
    }
    if (!check_epilogue()) {
        dbg_printf("line %d: epilogue has error\n", line);
        result = false;
    }

    // loop through the heap by calling helper function
    // this calls a helper function

    if (!heap_check(line)) {
        result = false;
    }

    /* counter counts the total number of free blocks in the heap
    and prints out the block that is being counted */
    word_t counter = 0;
    for (block = heap_start; get_size(block) > 0; block = find_next(block)) {
        if (!get_alloc(block)) {
            dbg_printf("free block in heap: %ld\n", (word_t)block);
            counter += 1;
        }
    }

    /* free_counter counts the total number of free blocks in the fre
    lists and prints out the block that is being counted */
    word_t free_counter = 0;
    for (word_t i = 0; i < BUCKET; i++) {

        if (free_start[i] != NULL) {

            free_counter += 1;
            dbg_printf("fre block in list %ld - index %ld\n",
                       (word_t)free_start[i], (word_t)i);

            for (block_t *block = free_start[i]->next; block != free_start[i];
                 block = block->next) {
                dbg_printf("free block in list %ld - index %ld\n",
                           (word_t)block, (word_t)i);
                free_counter += 1;
                if (!check_block(block, line)) {
                    result = false;
                }
            }
        }
    }

    // checking free_mini list
    for (block_t *block = free_mini; block != NULL; block = block->next_mini) {
        free_counter += 1;
        dbg_printf("free block in list %ld (mini block) - ", (word_t)block);
        dbg_printf("%ld\n", (word_t)block->next_mini);
    }

    if (counter != free_counter) {
        dbg_printf(
            "free block number is inconsistant with free block in heap\n");
        dbg_printf("free block number in heap is: %ld\n", counter);
        dbg_printf("free block number in list is: %ld\n", free_counter);
        result = false;
    }

    // check if block in correct segregated list
    for (word_t i = 0; i < BUCKET; i++) {
        block_t *block;
        if (free_start[i] != NULL) {

            if (get_list_index(get_size(free_start[i])) != i) {
                result = false;
                dbg_printf("free block start in incorrect segregated list");
            }

            for (block = free_start[i]->next; block != free_start[i];
                 block = block->next) {
                word_t size = get_size(block);
                if (get_list_index(size) != i) {
                    result = false;
                    dbg_printf("free block in incorrect segregated list");
                }
            }
        }
    }

    return result;
}

/**
 * @brief
 *
 * When mm_init is called, every global variables will be re-initialized
 * and there will be new heap allocated for the malloc and free trace
 *
 * @return Whether mm_init is successful or not
 */
bool mm_init(void) {
    dbg_printf("mm_init is called\n");
    // Create the initial empty heap
    word_t *start = (word_t *)(mem_sbrk(2 * wsize));
    dbg_assert(start != NULL);

    if (start == (void *)-1) {
        return false;
    }

    start[0] = pack_block(0, true, true, false); // Heap prologue (block footer)
    start[1] =
        pack_block(0, true, false, false); // Heap epilogue (block header)

    // Heap starts with first "block header", currently the epilogue
    // Free list starts with the first "block header", currently the epilogue
    heap_start = (block_t *)&(start[1]);

    // reinitialize global pointers to NULL
    for (word_t i = 0; i < BUCKET; i++) {
        free_start[i] = NULL;
    }
    free_mini = NULL;

    // Extend the empty heap with a free block of chunksize bytes
    dbg_printf("extend_heap\n");
    if (extend_heap(chunksize) == NULL) {
        return false;
    }

    return true;
}

/**
 * @brief
 *
 * This function allocates heap space for the malloc() call, and
 * extend heap when necessary
 *
 * This function calls find_fit() to search for usable free block in the
 * free list and return the pointer of the allocated space
 *
 * This function also calls split_block() to split large block if the
 * allocated space is much larger than needed.
 *
 * @param[in] size
 * @return Pointer of the allocated address
 */
void *malloc(size_t size) {
    dbg_printf("\n\nnew malloc!\n\n");
    dbg_printf("the malloced size is: %ld\n", size);

    size_t asize;      // Adjusted block size
    size_t extendsize; // Amount to extend heap if no fit is found
    block_t *block;
    void *bp = NULL;

    // Initialize heap if it isn't initialized
    if (heap_start == NULL) {
        mm_init();
    }

    // Ignore spurious request
    if (size == 0) {
        dbg_ensures(mm_checkheap(__LINE__));
        return bp;
    }

    // Adjust block size to include overhead and to meet alignment requirements
    asize = round_up(size + wsize, dsize);

    // Search the free list for a fit
    block = find_fit(asize);

    dbg_printf("find_fit returns: %ld\n", (word_t)block);

    // If no fit is found, request more memory, and then and place the block
    if (block == NULL) {
        // Always request at least chunksize
        extendsize = max(asize, chunksize);
        block = extend_heap(extendsize);
        // extend_heap returns an error
        if (block == NULL) {
            return bp;
        }
    }

    // The block should be marked as free
    dbg_assert(!get_alloc(block));

    // Mark block as allocated
    size_t block_size = get_size(block);
    bool prev_alloc = get_prev_alloc(block);
    bool prev_mini = get_prev_mini(block);
    write_block(block, block_size, true, prev_alloc, prev_mini);

    // Update info in the next block
    block_t *next = find_next(block);
    word_t next_large_size = get_size(next);
    if (next_large_size != 0) {
        bool next_alloc = get_alloc(next);
        bool cur_mini = (block_size == dsize);
        write_block(next, next_large_size, next_alloc, true, cur_mini);
    } else {
        bool cur_mini = (block_size == dsize);
        write_epilogue(next, true, cur_mini);
    }

    // Try to split the block if too large
    split_block(block, asize);
    dbg_printf("finished split_block in malloc\n");

    // Update epilogue info if malloced last block in heap
    block_t *next_block = find_next(block);
    word_t next_size = get_size(next_block);
    bool last_mini = (get_size(block) == dsize);

    if (next_size == 0) {
        write_epilogue(next_block, true, last_mini);
    }
    bp = header_to_payload(block);

    dbg_ensures(mm_checkheap(__LINE__));
    return bp;
}

/**
 * @brief
 *
 * This function represents the free call and frees the allocated
 * heap space for a pointer
 *
 * This function finds the block pointer of the allocated block,
 * writes the block as free in the heap and calls coalesce_block
 * to coalesce potential consecutive free blocks
 *
 * @param[in] bp The pointer to freed address
 */
void free(void *bp) {
    dbg_printf("\n\nthis is a free!\n\n");
    dbg_printf("freed pointer is: %ld\n", (word_t)bp);

    if (bp == NULL) {
        return;
    }

    block_t *block = payload_to_header(bp);
    size_t size = get_size(block);

    // The block should be marked as allocated
    dbg_assert(get_alloc(block));

    // Mark the block as free
    bool prev_alloc = get_prev_alloc(block);
    bool prev_mini = get_prev_mini(block);
    if (prev_alloc) {
        dbg_printf("In free, prev block is allocated\n");
    }
    write_block(block, size, false, prev_alloc, prev_mini);
    dbg_printf("block header: %ld\n", (word_t)block->header);
    dbg_printf("block next: %ld\n", (word_t)block->next);
    dbg_printf("block prev: %ld\n", (word_t)block->prev);
    insert_free(block);

    // Try to coalesce the block with its neighbors
    block = coalesce_block(block);

    dbg_ensures(mm_checkheap(__LINE__));
}

/**
 * @brief
 *
 * This function returns a pointer to an allocated region of at least size
 * bytes, for the following case:
 * 1) if ptr is NULL, the call is equivalent to malloc(size);
 * 2)if size is 0, the call is equivalent to free(ptr) followed by malloc(size)
 * and should return and should return NULL;
 * 3) if ptr is not NULL, the call is equivalent to free(ptr) followed by
 * malloc(size), except the contents of the new block will be the same as those
 * of the old block, up to the minimum of the old and new sizes
 *
 * @param[in] ptr A generic pointer
 * @param[in] size The size of realloc call
 * @return a pointer to an allocated region of at least size bytes
 */
void *realloc(void *ptr, size_t size) {
    block_t *block = payload_to_header(ptr);
    size_t copysize;
    void *newptr;

    // If size == 0, then free block and return NULL
    if (size == 0) {
        free(ptr);
        return NULL;
    }

    // If ptr is NULL, then equivalent to malloc
    if (ptr == NULL) {
        return malloc(size);
    }

    // Otherwise, proceed with reallocation
    newptr = malloc(size);

    // If malloc fails, the original block is left untouched
    if (newptr == NULL) {
        return NULL;
    }

    // Copy the old data
    copysize = get_payload_size(block); // gets size of old payload
    if (size < copysize) {
        copysize = size;
    }
    memcpy(newptr, ptr, copysize);

    // Free the old block
    free(ptr);

    return newptr;
}

/**
 * @brief
 *
 * This function takes in number of elements and size of each element
 * and allocate (number of elements * size of element) bytes through malloc
 *
 * This function initializes everything allocated as zero
 *
 * @param[in] elements Number of elements
 * @param[in] size Size of each element
 * @return pointer to the allocated space
 */
void *calloc(size_t elements, size_t size) {
    void *bp;
    size_t asize = elements * size;

    if (elements == 0) {
        return NULL;
    }
    if (asize / elements != size) {
        // Multiplication overflowed
        return NULL;
    }

    bp = malloc(asize);
    if (bp == NULL) {
        return NULL;
    }

    // Initialize all bits to 0
    memset(bp, 0, asize);

    return bp;
}
