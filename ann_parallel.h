/**********************************************************************************/
/* Copyright (c) 2023 Mark Seminatore                                             */
/* All rights reserved.                                                           */
/*                                                                                */
/* Permission is hereby granted, free of charge, to any person obtaining a copy   */
/* of this software and associated documentation files(the "Software"), to deal   */
/* in the Software without restriction, including without limitation the rights   */
/* to use, copy, modify, merge, publish, distribute, sublicense, and / or sell    */
/* copies of the Software, and to permit persons to whom the Software is          */
/* furnished to do so, subject to the following conditions:                       */
/*                                                                                */
/* The above copyright notice and this permission notice shall be included in all */
/* copies or substantial portions of the Software.                                */
/*                                                                                */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     */
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       */
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    */
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         */
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  */
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  */
/* SOFTWARE.                                                                      */
/**********************************************************************************/

#pragma once

#ifndef __ANN_PARALLEL_H
#define __ANN_PARALLEL_H

//================================================================================================
// Parallel loops (internal, implemented in tensor.c)
//
// ann_parallel_for() splits [0, count) into contiguous ranges and runs them on a small pool of
// worker threads, with the calling thread taking part. It runs the whole range on the calling
// thread instead when the range is small (count < 2 * grain), the thread count is 1, it is
// called from inside another parallel loop, or another thread is using the pool.
//
// Rule: range functions must not call BLAS unless ann_parallel_blas_ok() says they may. BLAS
// libraries run their own threads, so BLAS calls normally stay on the calling thread and libann's
// workers only run libann's own loops; the two never compete for the cores. Where BLAS threading
// can be set per thread (Accelerate on macOS 15+), range functions run with single-threaded
// BLAS, on the workers and on the calling thread alike, and may call it: many small GEMMs then
// run side by side instead of one at a time, each too small for BLAS's own threads.
//
// Build with ANN_NO_THREADS to compile the pool out; every loop then runs serially.
//================================================================================================

typedef void (*ann_range_func)(void *ctx, int begin, int end);

void ann_parallel_for(int count, int grain, ann_range_func fn, void *ctx);

// 1 if range functions may call BLAS (no BLAS, or it runs single-threaded inside them)
int  ann_parallel_blas_ok(void);

// Thread count used by ann_parallel_for (see ann_set_num_threads in ann.h)
void ann_parallel_set_threads(int threads);
int  ann_parallel_get_threads(void);

// The BLAS library tensor.c was built with, for ann_get_backend_info() (internal, implemented
// in tensor.c, where the BLAS header is included)
const char *ann_blas_backend_name(void);

#endif /* __ANN_PARALLEL_H */
