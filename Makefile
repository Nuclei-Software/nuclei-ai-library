TARGET = onnx_operators

ifeq ($(NUCLEI_SDK_ROOT),)
$(error "Please set correct NUCLEI_SDK_ROOT=/path/to/nuclei-sdk")
endif

SRCDIRS = . src test

INCDIRS = . src inc

CORE ?= nx900fd

ARCH_EXT ?= v_zfh_zvfh

LINKER_SCRIPT := evalsoc.ld

LDLIBS ?= -lm

COMMON_FLAGS := -O2

include $(NUCLEI_SDK_ROOT)/Build/Makefile.base
