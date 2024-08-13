TARGET = ailib_bench

NUCLEI_SDK_ROOT ?= ../../..

SRCDIRS = . src test

INCDIRS = . src inc

CORE ?= nx900fd

ARCH_EXT ?= v_zfh_zvfh

LINKER_SCRIPT := evalsoc.ld

LDLIBS ?= -lm

COMMON_FLAGS := -O2

ifneq ($(wildcard $(NUCLEI_SDK_ROOT)/Build/Makefile.base),)
include $(NUCLEI_SDK_ROOT)/Build/Makefile.base
else
$(error "Please set correct NUCLEI_SDK_ROOT=/path/to/nuclei-sdk")
endif
