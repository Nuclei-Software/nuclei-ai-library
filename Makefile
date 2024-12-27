TARGET = ailib_bench

NUCLEI_SDK_ROOT ?= ../../..

SRCDIRS = . src test

INCDIRS = . src inc

# Change to different CORE and ARCH
#= such as CORE=n900fd ARCH_EXT=_zfh_zvfh_zve32f
#= you can change it to bfloat16, eg. ARCH_EXT=v_zfh_zvfh_xxlvfbf
CORE ?= nx900fd
ARCH_EXT ?= v_zfh_zvfh

DOWNLOAD ?= ilm

TOOLCHAIN ?= nuclei_gnu

STDCLIB ?= newlib_small

# Change to different optimization level
# clang based TOOLCHAIN such as terapines, nuclei_llvm: https://clang.llvm.org/docs/CommandGuide/clang.html#cmdoption-O0
# gcc based such as nuclei_gnu: https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#index-O
OLV ?= O2

# Auto vectorization
# 0: Force disable
# 1: Force enable
# others: no changes, depends on compiler options
AUTOVEC ?= 0

# Bench flags which can be passed via make command
BENCH_FLAGS ?=

COMMON_FLAGS := $(BENCH_FLAGS) -$(OLV)

LINKER_SCRIPT ?= evalsoc.ld

LDLIBS := -lm

ifeq ($(AUTOVEC),1)
$(info Enable compiler auto vectorization!)
else ifeq ($(AUTOVEC),0)
$(info Disable compiler auto vectorization!)
endif
-include toolchain_$(TOOLCHAIN).mk

ifneq ($(wildcard $(NUCLEI_SDK_ROOT)/Build/Makefile.base),)
include $(NUCLEI_SDK_ROOT)/Build/Makefile.base
else
$(error "Please set correct NUCLEI_SDK_ROOT=/path/to/nuclei-sdk")
endif
