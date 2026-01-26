# get arch name
ARCH = $(shell uname -m)

# add Intel specific compiler flags
ifeq ($(ARCH), x86_64)
	CFLAGS += -mavx2 -mfma
endif

TARGET = mnist
OBJS = ann.o tensor.o
DEPS = ann.h tensor.h ann_config.h

# use no blas
#CFLAGS += -g -O2
#LFLAGS += -lm

# use cblas
# CFLAGS += -g -O2 -DUSE_BLAS -DCBLAS -I"/opt/cblas/include"
# LFLAGS += -L"/opt/cblas/lib" -lcblas -lm
# CFLAGS += -g -O2 -DUSE_BLAS -DCBLAS -I.
# LFLAGS += -L. -lcblas

# use openblas
CFLAGS += -g -O2 -DUSE_BLAS -I"/opt/OpenBLAS/include"
LFLAGS += -L/opt/OpenBLAS/lib/ -lopenblas

#-DMKL_ILP64  -m64  -I"${MKLROOT}/include"
#LFLAGS = ${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_tbb_thread.a ${MKLROOT}/lib/libmkl_core.a -L${TBBROOT}/lib -ltbb -lc++ -lpthread -lm -ldl

#  -DMKL_ILP64  -m64  -I"${MKLROOT}/include"
#  ${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_tbb_thread.a ${MKLROOT}/lib/libmkl_core.a -L${TBBROOT}/lib -ltbb -lc++ -lpthread -lm -ldl

all: mnist logic digit5x7 save_test save_test_binary blas_perf test_tensor test_network test_activations

$(TARGET):	$(OBJS) mnist.o
	$(CC) -o $@ $^ $(LFLAGS)

logic: $(OBJS) logic.o
	$(CC) -o $@ $^ $(LFLAGS)

digit5x7: $(OBJS) digit5x7.o
	$(CC) -o $@ $^ $(LFLAGS)

save_test: $(OBJS) save_test.o
	$(CC) -o $@ $^ $(LFLAGS)

save_test_binary: $(OBJS) save_test_binary.o
	$(CC) -o $@ $^ $(LFLAGS)

blas_perf: $(OBJS) blas_perf.o
	$(CC) -o $@ $^ $(LFLAGS)

test_tensor: $(OBJS) test_tensor.o testy/test_main.o
	$(CC) -o $@ $^ $(LFLAGS)

test_network: $(OBJS) test_network.o testy/test_main.o
	$(CC) -o $@ $^ $(LFLAGS)

test_activations: $(OBJS) test_activations.o testy/test_main.o
	$(CC) -o $@ $^ $(LFLAGS)

%.o: %.c $(DEPS)
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

clean:
	rm -f $(TARGET) $(OBJS) logic digit5x7 logic.o digit5x7.o mnist.o save_test.o save_test save_test_binary save_test_binary.o blas_perf.o test_tensor test_tensor.o test_network test_network.o test_activations test_activations.o testy/test_main.o

