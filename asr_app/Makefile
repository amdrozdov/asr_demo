default: app

THIRD_PARTY_DIR=./third_party
WHISPER_DIR=../whisper.cpp
CFLAGS   = -I. -O3 -DNDEBUG -std=c11   -fPIC -pthread
CXXFLAGS = -I. -O3 -DNDEBUG -std=c++11 -fPIC -pthread

CXX=g++-8
CC=gcc-8
CCV := $(shell $(CC) --version | head -n 1)
CXXV := $(shell $(CXX) --version | head -n 1)
SRC_COMMON     = $(THIRD_PARTY_DIR)/common.cpp $(THIRD_PARTY_DIR)/common-ggml.cpp
SRC_COMMON_SDL = $(THIRD_PARTY_DIR)/common-sdl.cpp
INC_DIR = $(THIRD_PARTY_DIR)/
CC_SDL=`sdl2-config --cflags --libs`
WHISPER_OBJ = $(WHISPER_DIR)/whisper.o
GGML_OBJ = $(WHISPER_DIR)/ggml.o

app:
	$(CXX) $(CXXFLAGS) asr_app.cpp main.cpp $(SRC_COMMON) $(SRC_COMMON_SDL) $(GGML_OBJ) $(WHISPER_OBJ) -I$(INC_DIR) -lcurl -o asr_app $(CC_SDL)
clean:
	rm asr_app
