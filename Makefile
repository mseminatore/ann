TARGET = ann
OBJS = ann.o tensor.o
DEPS = ann.h tensor.h
CFLAGS = -g -mavx -O3

all: ann logic test5x7 pima

$(TARGET):	$(OBJS) main.o
	$(CC) $(CFLAGS) -o $@ $^

logic: $(OBJS) logic.o
	$(CC) $(CFLAGS) -o $@ $^

test5x7: $(OBJS) test5x7.o
	$(CC) $(CFLAGS) -o $@ $^

pima: $(OBJS) pima.o
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c $(DEPS)
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

clean:
	rm $(TARGET) $(OBJS) logic test5x7 logic.o test5x7.o main.o

