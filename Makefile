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
CFLAGS += -g -O2 -DUSE_BLAS -I"/opt/OpenBLAS/include"
LFLAGS += -L/opt/OpenBLAS/lib/ -lopenblas

#-DMKL_ILP64  -m64  -I"${MKLROOT}/include"
#LFLAGS = ${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_tbb_thread.a ${MKLROOT}/lib/libmkl_core.a -L${TBBROOT}/lib -ltbb -lc++ -lpthread -lm -ldl

#  -DMKL_ILP64  -m64  -I"${MKLROOT}/include"
#  ${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_tbb_thread.a ${MKLROOT}/lib/libmkl_core.a -L${TBBROOT}/lib -ltbb -lc++ -lpthread -lm -ldl

all: $(LIBANN) mnist logic digit5x7 save_test save_test_binary blas_perf test_tensor test_network test_activations test_loss_functions test_save_load test_optimizers test_forward_pass test_training_convergence test_onnx_export test_hypertune test_json

# build the static library
$(LIBANN): $(LIB_OBJS)
	$(AR) rcs $@ $^

$(TARGET): $(LIBANN) mnist.o
	$(CC) -o $@ mnist.o $(LIBANN) $(LFLAGS)

logic: $(LIBANN) logic.o
	$(CC) -o $@ logic.o $(LIBANN) $(LFLAGS)

digit5x7: $(LIBANN) digit5x7.o
	$(CC) -o $@ digit5x7.o $(LIBANN) $(LFLAGS)

save_test: $(LIBANN) save_test.o
	$(CC) -o $@ save_test.o $(LIBANN) $(LFLAGS)

save_test_binary: $(LIBANN) save_test_binary.o
	$(CC) -o $@ save_test_binary.o $(LIBANN) $(LFLAGS)

blas_perf: $(LIBANN) blas_perf.o
	$(CC) -o $@ blas_perf.o $(LIBANN) $(LFLAGS)

test_tensor: $(LIBANN) test_tensor.o testy/test_main.o
	$(CC) -o $@ test_tensor.o testy/test_main.o $(LIBANN) $(LFLAGS)

test_network: $(LIBANN) test_network.o testy/test_main.o
	$(CC) -o $@ test_network.o testy/test_main.o $(LIBANN) $(LFLAGS)

test_activations: $(LIBANN) test_activations.o testy/test_main.o
	$(CC) -o $@ test_activations.o testy/test_main.o $(LIBANN) $(LFLAGS)

test_loss_functions: $(LIBANN) test_loss_functions.o testy/test_main.o
	$(CC) -o $@ test_loss_functions.o testy/test_main.o $(LIBANN) $(LFLAGS)

test_save_load: $(LIBANN) test_save_load.o testy/test_main.o
	$(CC) -o $@ test_save_load.o testy/test_main.o $(LIBANN) $(LFLAGS)

test_optimizers: $(LIBANN) test_optimizers.o testy/test_main.o
	$(CC) -o $@ test_optimizers.o testy/test_main.o $(LIBANN) $(LFLAGS)

test_forward_pass: $(LIBANN) test_forward_pass.o testy/test_main.o
	$(CC) -o $@ test_forward_pass.o testy/test_main.o $(LIBANN) $(LFLAGS)

test_training_convergence: $(LIBANN) test_training_convergence.o testy/test_main.o
	$(CC) -o $@ test_training_convergence.o testy/test_main.o $(LIBANN) $(LFLAGS)

test_onnx_export: $(LIBANN) test_onnx_export.o testy/test_main.o
	$(CC) -o $@ test_onnx_export.o testy/test_main.o $(LIBANN) $(LFLAGS)

test_hypertune: $(LIBANN) test_hypertune.o testy/test_main.o
	$(CC) -o $@ test_hypertune.o testy/test_main.o $(LIBANN) $(LFLAGS)

test_json: $(LIBANN) test_json.o testy/test_main.o
	$(CC) -o $@ test_json.o testy/test_main.o $(LIBANN) $(LFLAGS)

%.o: %.c $(DEPS)
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

clean:
	rm -f $(LIBANN) $(LIB_OBJS) $(TARGET) logic digit5x7 logic.o digit5x7.o mnist.o save_test.o save_test save_test_binary save_test_binary.o blas_perf blas_perf.o test_tensor test_tensor.o test_network test_network.o test_activations test_activations.o test_loss_functions test_loss_functions.o test_save_load test_save_load.o test_optimizers test_optimizers.o test_forward_pass test_forward_pass.o test_training_convergence test_training_convergence.o test_onnx_export test_onnx_export.o test_hypertune test_hypertune.o test_json test_json.o testy/test_main.o

