#OPENCV = /usr/local/opencv
OPENCV = /usr/local/opencv-3.0.0

CC = g++
CPPFLAGS = -O2 -W -Wall -I$(OPENCV)/include
LDLIBS = -L$(OPENCV)/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs

BIN = imdiff
all: $(BIN)

imdiff: imdiff.o ImageIOpfm.o

clean:
	rm -f $(BIN) *.o core*
