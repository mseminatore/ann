TARGET = ann
OBJS = main.o ann.o
DEPS = ann.h
CFLAGS = -g

$(TARGET):	$(OBJS) $(DEPS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

clean:
	rm $(TARGET) $(OBJS)

