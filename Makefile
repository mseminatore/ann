# get arch name
ARCH = $(shell uname -m)

# add Intel specific compiler flags
ifeq ($(ARCH), x86_64)
	CFLAGS += -mavx2 -mfma
endif

TARGET = mnist
LIB_OBJS = ann.o tensor.o json.o ann_hypertune.o
LIBANN = libann.a
DEPS = ann.h tensor.h ann_config.h json.h ann_hypertune.h

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

#-DMKL_ILP64  -m64  -I"${MKLROOT}/include"
#LFLAGS = ${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_tbb_thread.a ${MKLROOT}/lib/libmkl_core.a -L${TBBROOT}/lib -ltbb -lc++ -lpthread -lm -ldl

all: $(LIBANN) mnist mnist_hypertune logic digit5x7 save_test save_test_binary blas_perf test_tensor test_network test_activations test_loss_functions test_save_load test_optimizers test_forward_pass test_training_convergence test_onnx_export test_hypertune test_json

# build the static library
$(LIBANN): $(LIB_OBJS)
	$(AR) rcs $@ $^

# build the shared library
LIBANN_SO = libann.so
SHARED_OBJS = ann.pic.o tensor.pic.o json.pic.o ann_hypertune.pic.o

%.pic.o: %.c $(DEPS)
	$(CC) -c -fPIC $(CFLAGS) -o $@ $<

shared: $(LIBANN_SO)

$(LIBANN_SO): $(SHARED_OBJS)
	$(CC) -shared -o $@ $^ $(LFLAGS)

# examples
$(TARGET): $(LIBANN) examples/mnist.o
	$(CC) -o $@ examples/mnist.o $(LIBANN) $(LFLAGS)

mnist_hypertune: $(LIBANN) examples/mnist_hypertune.o
	$(CC) -o $@ examples/mnist_hypertune.o $(LIBANN) $(LFLAGS)

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

# implicit rules
%.o: %.c $(DEPS)
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

tests/%.o: tests/%.c $(DEPS)
	$(CC) -c $(CFLAGS) $(CPPFLAGS) -I. $< -o $@

examples/%.o: examples/%.c $(DEPS)
	$(CC) -c $(CFLAGS) $(CPPFLAGS) -I. $< -o $@

clean:
	rm -f $(LIBANN) $(LIBANN_SO) $(LIB_OBJS) $(SHARED_OBJS) $(TARGET) logic digit5x7 mnist_hypertune save_test save_test_binary blas_perf test_tensor test_network test_activations test_loss_functions test_save_load test_optimizers test_forward_pass test_training_convergence test_onnx_export test_hypertune test_json testy/test_main.o
	rm -f tests/*.o examples/*.o
