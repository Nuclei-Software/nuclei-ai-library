TARGET = onnx_operators

ifeq ($(NUCLEI_SDK_ROOT),)
$(error "Please set correct NUCLEI_SDK_ROOT=/path/to/nuclei-sdk")
endif

SRCDIRS = . src test

INCDIRS = . src inc

ARCH_EXT ?= v_zfh_zvfh

LDLIBS ?= -lm

COMMON_FLAGS := -O2

include $(NUCLEI_SDK_ROOT)/Build/Makefile.base
