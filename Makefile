TARGET = ann
OBJS = main.o ann.o tensor.o
DEPS = ann.h tensor.h
CFLAGS = -g

$(TARGET):	$(OBJS) $(DEPS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

clean:
	rm $(TARGET) $(OBJS)

