#include <stdlib.h>
#include <cblas.h>

#define SIZE (1 << 10)
#define LOOPS 1000

__attribute__((aligned(32))) float a[SIZE];
__attribute__((aligned(32))) float b[SIZE];
__attribute__((aligned(32))) float c[SIZE];

//
void init_rand(float *array, int count)
{
	for (int i = 0; i < count; i++)
	{
		array[i] = 1.0f;    //(float)rand() / (float)RAND_MAX;
	}
}

int main(int argc, char *argv[])
{
    init_rand(a, SIZE);
	init_rand(b, SIZE);

    a[1023] = 100.0f;

    cblas_sgemv(CblasColMajor, CblasNoTrans, 1, SIZE, 1.0f, a, 1, b, 1, 0.0f, c, 1);

    float r = cblas_sdot(SIZE, a, 1, b, 1);

    printf("Dot product is: %f\n", r);

    int i = cblas_isamax(SIZE, a, 1);

    printf("Max element is at: %d\n", i);

    return 0;
}