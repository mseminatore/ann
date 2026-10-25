# get arch name
ARCH = $(shell uname -m)

# add Intel specific compiler flags
ifeq ($(ARCH), x86_64)
	CFLAGS += -mavx2 -mfma
endif

TARGET = mnist
LIB_OBJS = ann.o tensor.o json.o ann_hypertune.o ann_quantize.o
LIBANN = libann.a
DEPS = ann.h ann_rt.h tensor.h ann_config.h json.h ann_hypertune.h

# use no blas
#CFLAGS += -g -O2
#LFLAGS += -lm

# use cblas (requires C11 atomics)
# CFLAGS += -std=c11 -g -O2 -DUSE_BLAS -DCBLAS -I"/opt/cblas/include"
# LFLAGS += -L"/opt/cblas/lib" -lcblas -lm
# CFLAGS += -std=c11 -g -O2 -DUSE_BLAS -DCBLAS -I.
# LFLAGS += -L. -lcblas

# use openblas
CFLAGS += -g -O2 -I. -DUSE_BLAS -I"/opt/OpenBLAS/include"
LFLAGS += -L/opt/OpenBLAS/lib/ -lopenblas

# ann_parallel_for's worker threads
CFLAGS += -pthread
LFLAGS += -pthread

#-DMKL_ILP64  -m64  -I"${MKLROOT}/include"
#LFLAGS = ${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_tbb_thread.a ${MKLROOT}/lib/libmkl_core.a -L${TBBROOT}/lib -ltbb -lc++ -lpthread -lm -ldl

all: $(LIBANN) mnist fashion_cnn fashion_dae export_model mnist_hypertune_tpe mnist_hypertune_grid mnist_hypertune_random mnist_hypertune_bayes logic digit5x7 save_test save_test_binary blas_perf test_tensor test_network test_activations test_loss_functions test_save_load test_optimizers test_forward_pass test_training_convergence test_onnx_export test_hypertune test_json test_gradients test_comprehensive_callbacks test_training_options test_conv test_batchnorm test_embedding test_threads test_noise test_quantize

# build the static library
$(LIBANN): $(LIB_OBJS)
	$(AR) rcs $@ $^

# build the shared library
LIBANN_SO = libann.so
SHARED_OBJS = ann.pic.o tensor.pic.o json.pic.o ann_hypertune.pic.o ann_quantize.pic.o

# export only ANN_API-marked functions
%.pic.o: %.c $(DEPS)
	$(CC) -c -fPIC -fvisibility=hidden -DANN_BUILDING_DLL $(CFLAGS) -o $@ $<

shared: $(LIBANN_SO)

$(LIBANN_SO): $(SHARED_OBJS)
	$(CC) -shared -o $@ $^ $(LFLAGS)

# examples
$(TARGET): $(LIBANN) examples/mnist.o
	$(CC) -o $@ examples/mnist.o $(LIBANN) $(LFLAGS)

fashion_cnn: $(LIBANN) examples/fashion_cnn.o
	$(CC) -o $@ examples/fashion_cnn.o $(LIBANN) $(LFLAGS)

fashion_dae: $(LIBANN) examples/fashion_dae.o
	$(CC) -o $@ examples/fashion_dae.o $(LIBANN) $(LFLAGS)

export_model: $(LIBANN) examples/export_model.o
	$(CC) -o $@ examples/export_model.o $(LIBANN) $(LFLAGS)

# needs fashion_q.h and fashion_f.h from export_model; not part of 'all'
run_exported: examples/run_exported.c fashion_q.h fashion_f.h ann_rt.h
	$(CC) -O2 -I. -o $@ examples/run_exported.c -lm

mnist_hypertune_tpe: $(LIBANN) examples/mnist_hypertune_tpe.o
	$(CC) -o $@ examples/mnist_hypertune_tpe.o $(LIBANN) $(LFLAGS)

mnist_hypertune_grid: $(LIBANN) examples/mnist_hypertune_grid.o
	$(CC) -o $@ examples/mnist_hypertune_grid.o $(LIBANN) $(LFLAGS)

mnist_hypertune_random: $(LIBANN) examples/mnist_hypertune_random.o
	$(CC) -o $@ examples/mnist_hypertune_random.o $(LIBANN) $(LFLAGS)

mnist_hypertune_bayes: $(LIBANN) examples/mnist_hypertune_bayes.o
	$(CC) -o $@ examples/mnist_hypertune_bayes.o $(LIBANN) $(LFLAGS)

logic: $(LIBANN) examples/logic.o
	$(CC) -o $@ examples/logic.o $(LIBANN) $(LFLAGS)

digit5x7: $(LIBANN) examples/digit5x7.o
	$(CC) -o $@ examples/digit5x7.o $(LIBANN) $(LFLAGS)

save_test: $(LIBANN) examples/save_test.o
	$(CC) -o $@ examples/save_test.o $(LIBANN) $(LFLAGS)

save_test_binary: $(LIBANN) examples/save_test_binary.o
	$(CC) -o $@ examples/save_test_binary.o $(LIBANN) $(LFLAGS)

blas_perf: $(LIBANN) examples/blas_perf.o
	$(CC) -o $@ examples/blas_perf.o $(LIBANN) $(LFLAGS)

# tests
test_tensor: $(LIBANN) tests/test_tensor.o testy/test_main.o
	$(CC) -o $@ tests/test_tensor.o testy/test_main.o $(LIBANN) $(LFLAGS)

test_network: $(LIBANN) tests/test_network.o testy/test_main.o
	$(CC) -o $@ tests/test_network.o testy/test_main.o $(LIBANN) $(LFLAGS)

test_activations: $(LIBANN) tests/test_activations.o testy/test_main.o
	$(CC) -o $@ tests/test_activations.o testy/test_main.o $(LIBANN) $(LFLAGS)

test_loss_functions: $(LIBANN) tests/test_loss_functions.o testy/test_main.o
	$(CC) -o $@ tests/test_loss_functions.o testy/test_main.o $(LIBANN) $(LFLAGS)

test_save_load: $(LIBANN) tests/test_save_load.o testy/test_main.o
	$(CC) -o $@ tests/test_save_load.o testy/test_main.o $(LIBANN) $(LFLAGS)

test_optimizers: $(LIBANN) tests/test_optimizers.o testy/test_main.o
	$(CC) -o $@ tests/test_optimizers.o testy/test_main.o $(LIBANN) $(LFLAGS)

test_forward_pass: $(LIBANN) tests/test_forward_pass.o testy/test_main.o
	$(CC) -o $@ tests/test_forward_pass.o testy/test_main.o $(LIBANN) $(LFLAGS)

test_training_convergence: $(LIBANN) tests/test_training_convergence.o testy/test_main.o
	$(CC) -o $@ tests/test_training_convergence.o testy/test_main.o $(LIBANN) $(LFLAGS)

test_onnx_export: $(LIBANN) tests/test_onnx_export.o testy/test_main.o
	$(CC) -o $@ tests/test_onnx_export.o testy/test_main.o $(LIBANN) $(LFLAGS)

test_hypertune: $(LIBANN) tests/test_hypertune.o testy/test_main.o
	$(CC) -o $@ tests/test_hypertune.o testy/test_main.o $(LIBANN) $(LFLAGS)

test_json: $(LIBANN) tests/test_json.o testy/test_main.o
	$(CC) -o $@ tests/test_json.o testy/test_main.o $(LIBANN) $(LFLAGS)

test_gradients: $(LIBANN) tests/test_gradients.o testy/test_main.o
	$(CC) -o $@ tests/test_gradients.o testy/test_main.o $(LIBANN) $(LFLAGS)

test_comprehensive_callbacks: $(LIBANN) tests/test_comprehensive_callbacks.o testy/test_main.o
	$(CC) -o $@ tests/test_comprehensive_callbacks.o testy/test_main.o $(LIBANN) $(LFLAGS)

test_training_options: $(LIBANN) tests/test_training_options.o testy/test_main.o
	$(CC) -o $@ tests/test_training_options.o testy/test_main.o $(LIBANN) $(LFLAGS)

test_conv: $(LIBANN) tests/test_conv.o testy/test_main.o
	$(CC) -o $@ tests/test_conv.o testy/test_main.o $(LIBANN) $(LFLAGS)

test_batchnorm: $(LIBANN) tests/test_batchnorm.o testy/test_main.o
	$(CC) -o $@ tests/test_batchnorm.o testy/test_main.o $(LIBANN) $(LFLAGS)

test_embedding: $(LIBANN) tests/test_embedding.o testy/test_main.o
	$(CC) -o $@ tests/test_embedding.o testy/test_main.o $(LIBANN) $(LFLAGS)

test_threads: $(LIBANN) tests/test_threads.o testy/test_main.o
	$(CC) -o $@ tests/test_threads.o testy/test_main.o $(LIBANN) $(LFLAGS)

test_noise: $(LIBANN) tests/test_noise.o testy/test_main.o
	$(CC) -o $@ tests/test_noise.o testy/test_main.o $(LIBANN) $(LFLAGS)

test_quantize: $(LIBANN) tests/test_quantize.o testy/test_main.o
	$(CC) -o $@ tests/test_quantize.o testy/test_main.o $(LIBANN) $(LFLAGS)

# implicit rules
%.o: %.c $(DEPS)
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

tests/%.o: tests/%.c $(DEPS)
	$(CC) -c $(CFLAGS) $(CPPFLAGS) -I. $< -o $@

examples/%.o: examples/%.c $(DEPS)
	$(CC) -c $(CFLAGS) $(CPPFLAGS) -I. $< -o $@

clean:
	rm -f $(LIBANN) $(LIBANN_SO) $(LIB_OBJS) $(SHARED_OBJS) $(TARGET) fashion_cnn fashion_dae export_model run_exported logic digit5x7 mnist_hypertune_tpe mnist_hypertune_grid mnist_hypertune_random mnist_hypertune_bayes save_test save_test_binary blas_perf test_tensor test_network test_activations test_loss_functions test_save_load test_optimizers test_forward_pass test_training_convergence test_onnx_export test_hypertune test_json test_gradients test_comprehensive_callbacks test_training_options test_conv test_batchnorm test_embedding test_threads test_noise test_quantize testy/test_main.o
	rm -f tests/*.o examples/*.o
