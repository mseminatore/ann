//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#if defined(_linux_) || defined(__linux__) || defined(__linux) || defined(__gnu_linux__)
#define _POSIX_C_SOURCE 199309L
#endif

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <cblas.h>

#ifdef WIN32
#   include <Windows.h>
#endif

#define MAX_SIZE 1024

float x[MAX_SIZE], y[MAX_SIZE];
float a[MAX_SIZE * MAX_SIZE], b[MAX_SIZE * MAX_SIZE], c[MAX_SIZE * MAX_SIZE];

struct timer
{
#ifdef WIN32
    LARGE_INTEGER t;
#else
    struct timespec t;
#endif

};

//------------------------------------------------------
//
//------------------------------------------------------
void timer_get_time(struct timer* t)
{
#ifdef WIN32
    QueryPerformanceCounter(&t->t);
#else
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t->t);
#endif

}

//------------------------------------------------------
//
//------------------------------------------------------
float timer_get_delta(struct timer *t1, struct timer *t2)
{
    float dt;

#ifdef WIN32
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);

    dt = (t2->t.QuadPart - t1->t.QuadPart) / (float)freq.QuadPart;

#else
    int seconds = (int)(t2->t.tv_sec - t1->t.tv_sec);
    long long ns = t2->t.tv_nsec - t1->t.tv_nsec;
    dt = (float)seconds + (float)ns / (1000000000);
#endif

    return dt;
}

//------------------------------------------------------
//
//------------------------------------------------------
void test_gemm()
{
    struct timer t1, t2;
    CBLAS_INDEX m, n, k;
    float dt;

    printf("Testing performance of cblas_sgemm()\n\n");

    for (int i = 4; i <= MAX_SIZE; i <<= 1)
    {
        m = n = k = i;

        timer_get_time(&t1);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, m, b, k, 1.0f, c, k);

        timer_get_time(&t2);

        dt = timer_get_delta(&t1, &t2);

        printf("%4d: %5.2f GFlops in %5.2fs\n", i, (float)2 * m * n * k / 1000000000 / dt, dt);
    }
}

//------------------------------------------------------
//
//------------------------------------------------------
void test_ger()
{
    struct timer t1, t2;
    float dt;

    printf("Testing performance of cblas_sger()\n\n");

    CBLAS_INDEX m = MAX_SIZE, n = MAX_SIZE;

    for (int i = 2; i <= MAX_SIZE; i <<= 1)
    {
        m = n = i;

        timer_get_time(&t1);

        cblas_sger(CblasRowMajor, m, n, 1.0f, x, 1, y, 1, a, m);

        timer_get_time(&t2);

        dt = timer_get_delta(&t1, &t2);

        printf("%4d: %5.2f GFlops in %5.2fs\n", i, (float)2 * m * n / 1000000000 / dt, dt);
    }
}

//------------------------------------------------------
//
//------------------------------------------------------
int main(int argc, char *argv[])
{
#ifdef CBLAS
	cblas_init(CBLAS_DEFAULT_THREADS);
    cblas_print_configuration();
#else
    printf( "%s\n", openblas_get_config());
    printf("    CPU uArch: %s\n", openblas_get_corename());
    printf("Cores/Threads: %d/%d\n\n", openblas_get_num_procs(), openblas_get_num_threads());
#endif

    test_gemm();

	return 0;
}