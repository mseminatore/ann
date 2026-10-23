//-----------------------------------------------------------
// Unloading the shared library after its thread pool has started. pyann loads
// libann with cffi, which unloads it at interpreter exit; on Windows that
// unload once deadlocked joining the workers under the loader lock. The test
// loads a shared copy of libann (ANN_UNLOAD_DLL), runs a batch through a layer
// large enough to start the pool, and unloads it on a helper thread with a
// time limit.
//-----------------------------------------------------------

// clock_gettime() and pthread_cond_timedwait() under -std=c11
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#	define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h>
#include <stdlib.h>
#include "ann.h"
#include "testy/test.h"

#ifdef _WIN32
#	include <windows.h>
typedef HMODULE lib_handle;
#	define lib_open(path)		LoadLibraryA(path)
#	define lib_symbol(h, name)	((void *)GetProcAddress(h, name))
#	define lib_close(h)			FreeLibrary(h)
#else
#	include <dlfcn.h>
#	include <pthread.h>
#	include <time.h>
#	include <errno.h>
#	include <unistd.h>
typedef void *lib_handle;
#	define lib_open(path)		dlopen(path, RTLD_NOW | RTLD_LOCAL)
#	define lib_symbol(h, name)	dlsym(h, name)
#	define lib_close(h)			dlclose(h)
#endif

#define ROWS	256
#define INPUTS	64
#define HIDDEN	512
#define OUTPUTS	4

// seconds to wait for the unload before calling it hung
#define UNLOAD_LIMIT 20

typedef PNetwork (*make_network_fn)(Optimizer_type, Loss_type);
typedef int (*add_layer_fn)(PNetwork, int, Layer_type, Activation_type);
typedef int (*predict_batch_fn)(const PNetwork, const real *, real *, int);
typedef void (*free_network_fn)(PNetwork);
typedef void (*set_num_threads_fn)(int);

static lib_handle g_lib;

//------------------------------
// predict a batch through the loaded library, using several threads
//------------------------------
static int run_parallel_predict(void)
{
	make_network_fn make_network = (make_network_fn)lib_symbol(g_lib, "ann_make_network");
	add_layer_fn add_layer = (add_layer_fn)lib_symbol(g_lib, "ann_add_layer");
	predict_batch_fn predict_batch = (predict_batch_fn)lib_symbol(g_lib, "ann_predict_batch");
	free_network_fn free_network = (free_network_fn)lib_symbol(g_lib, "ann_free_network");
	set_num_threads_fn set_num_threads = (set_num_threads_fn)lib_symbol(g_lib, "ann_set_num_threads");
	if (!make_network || !add_layer || !predict_batch || !free_network || !set_num_threads)
		return 0;

	// at least two threads, so a single-core runner still starts a worker
	set_num_threads(4);

	PNetwork net = make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
	if (!net)
		return 0;
	add_layer(net, INPUTS, LAYER_INPUT, ACTIVATION_NULL);
	add_layer(net, HIDDEN, LAYER_HIDDEN, ACTIVATION_SIGMOID);
	add_layer(net, OUTPUTS, LAYER_OUTPUT, ACTIVATION_SOFTMAX);

	real *inputs = malloc(ROWS * INPUTS * sizeof(real));
	real *outputs = malloc(ROWS * OUTPUTS * sizeof(real));
	for (int i = 0; i < ROWS * INPUTS; i++)
		inputs[i] = (real)(i % 17) / 17.0f;

	int ok = predict_batch(net, inputs, outputs, ROWS) == ERR_OK;

	free(inputs);
	free(outputs);
	free_network(net);
	return ok;
}

//------------------------------
// unload the library on a helper thread; 1 if it finished within the limit
//------------------------------
#ifdef _WIN32
static DWORD WINAPI unload_thread(LPVOID arg)
{
	(void)arg;
	lib_close(g_lib);
	return 0;
}

static int unload_within_limit(void)
{
	HANDLE h = CreateThread(NULL, 0, unload_thread, NULL, 0, NULL);
	if (!h)
		return 0;
	int done = WaitForSingleObject(h, UNLOAD_LIMIT * 1000) == WAIT_OBJECT_0;
	CloseHandle(h);
	return done;
}
#else
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_cond = PTHREAD_COND_INITIALIZER;
static int g_unloaded;

static void *unload_thread(void *arg)
{
	(void)arg;
	lib_close(g_lib);
	pthread_mutex_lock(&g_lock);
	g_unloaded = 1;
	pthread_cond_signal(&g_cond);
	pthread_mutex_unlock(&g_lock);
	return NULL;
}

static int unload_within_limit(void)
{
	pthread_t t;
	if (pthread_create(&t, NULL, unload_thread, NULL) != 0)
		return 0;

	struct timespec deadline;
	clock_gettime(CLOCK_REALTIME, &deadline);
	deadline.tv_sec += UNLOAD_LIMIT;

	pthread_mutex_lock(&g_lock);
	int err = 0;
	while (!g_unloaded && err != ETIMEDOUT)
		err = pthread_cond_timedwait(&g_cond, &g_lock, &deadline);
	int done = g_unloaded;
	pthread_mutex_unlock(&g_lock);

	if (done)
		pthread_join(t, NULL);
	return done;
}
#endif

//------------------------------
// leave without running exit handlers or DLL notifications, which would block
// on the stuck unload
//------------------------------
static void exit_now(int code)
{
	fflush(stdout);
#ifdef _WIN32
	TerminateProcess(GetCurrentProcess(), code);
#else
	_exit(code);
#endif
}

void test_main(int argc, char *argv[])
{
	(void)argc; (void)argv;

	MODULE("Shared library unload");

	SUITE("Unload after the thread pool starts");
	{
		g_lib = lib_open(ANN_UNLOAD_DLL);
		TESTEX("Shared library loads", g_lib != NULL);
		if (!g_lib)
			return;

		TESTEX("Batch prediction runs on several threads", run_parallel_predict());

		int unloaded = unload_within_limit();
		TESTEX("Unloading the library does not hang", unloaded);
		if (!unloaded)
			exit_now(1);
	}
}
