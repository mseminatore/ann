TARGET = ann
OBJS = ann.o tensor.o
DEPS = ann.h tensor.h
CFLAGS = -g -mavx -O3 -I"/opt/OpenBLAS/include"

LFLAGS = -L/opt/OpenBLAS/lib -lopenblas
#-DMKL_ILP64  -m64  -I"${MKLROOT}/include"
#LFLAGS = ${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_tbb_thread.a ${MKLROOT}/lib/libmkl_core.a -L${TBBROOT}/lib -ltbb -lc++ -lpthread -lm -ldl

#  -DMKL_ILP64  -m64  -I"${MKLROOT}/include"
#  ${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_tbb_thread.a ${MKLROOT}/lib/libmkl_core.a -L${TBBROOT}/lib -ltbb -lc++ -lpthread -lm -ldl

all: ann logic test5x7 pima save_test

$(TARGET):	$(OBJS) main.o
	$(CC) $(LFLAGS) -o $@ $^

logic: $(OBJS) logic.o
	$(CC) $(LFLAGS) -o $@ $^

test5x7: $(OBJS) test5x7.o
	$(CC) $(LFLAGS) -o $@ $^

pima: $(OBJS) pima.o
	$(CC) $(LFLAGS) -o $@ $^

save_test: $(OBJS) save_test.o
	$(CC) $(LFLAGS) -o $@ $^

%.o: %.c $(DEPS)
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

clean:
	rm $(TARGET) $(OBJS) logic test5x7 logic.o test5x7.o main.o save_test.o pima.o pima save_test

