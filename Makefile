# get arch name
ARCH = $(shell uname -m)

# add Intel specific compiler flags
ifeq ($(ARCH), x86_64)
	CFLAGS += -mavx2 -mfma
endif

TARGET = mnist
OBJS = ann.o tensor.o
DEPS = ann.h tensor.h ann_config.h
CFLAGS += -g -O3 -DUSE_BLAS -I"/opt/OpenBLAS/include"

LFLAGS += -L/opt/OpenBLAS/lib/ -lopenblas
#-DMKL_ILP64  -m64  -I"${MKLROOT}/include"
#LFLAGS = ${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_tbb_thread.a ${MKLROOT}/lib/libmkl_core.a -L${TBBROOT}/lib -ltbb -lc++ -lpthread -lm -ldl

#  -DMKL_ILP64  -m64  -I"${MKLROOT}/include"
#  ${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_tbb_thread.a ${MKLROOT}/lib/libmkl_core.a -L${TBBROOT}/lib -ltbb -lc++ -lpthread -lm -ldl

all: mnist logic digit5x7 save_test save_test_binary

$(TARGET):	$(OBJS) mnist.o
	$(CC) $(LFLAGS) -o $@ $^

logic: $(OBJS) logic.o
	$(CC) $(LFLAGS) -o $@ $^

digit5x7: $(OBJS) digit5x7.o
	$(CC) $(LFLAGS) -o $@ $^

save_test: $(OBJS) save_test.o
	$(CC) $(LFLAGS) -o $@ $^

save_test_binary: $(OBJS) save_test_binary.o
	$(CC) $(LFLAGS) -o $@ $^

%.o: %.c $(DEPS)
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

clean:
	rm $(TARGET) $(OBJS) logic digit5x7 logic.o digit5x7.o mnist.o save_test.o save_test save_test_binary save_test_binary.o

