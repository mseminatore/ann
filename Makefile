TARGET = ann
OBJS = main.o ann.o tensor.o
DEPS = ann.h tensor.h
CFLAGS = -g -mavx

$(TARGET):	$(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

%.o: %.c $(DEPS)
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

clean:
	rm $(TARGET) $(OBJS)

