# Nuclei AI Library

Nuclei AI Library is a set of ONNX AI operators optimized for Nuclei RISC-V Processors which support RISC-V Vector Instruction Set.

We implemented the ONNX Operators in pure c code, and also provided RISC-V Vector optimized implementation, see source code located in `src` for details.

We also provided test code to evaluate the ONNX operators implemention, which can be evaluated with Nuclei SDK.

## Supported ONNX Operators

> Some operator implementation may only support subset of the ONNX operator.

**VPU Lite**: VPU Lite is a lightweight VPU implementation, which **don't support** following features in whole or in part:

- segment load/store
- vslide/vgather/vcompress
- ELEN=64

In the chart below, `VPU Lite Compatibility` illustrates the degree to which each operator is compatible with VPU Lite. The symbol `√` indicates that the operator is fully compatible with VPU Lite. In instances where compatibility is not achieved, the chart will illustrates the reasons why the operator is not compatible.

| Operator           | VPU Lite compatibility | FP32 | FP16 | BF16 | FP8 | INT32 | INT8 | INT4 | Boolean |
| --                 | --   | --   | --   | --   | --  | --    | --   | --   | --      |
| Abs                | √    | √    | √    | ×    | ×   | √     |  √   | ×    |   |
| Add                | √    | √    | √    | ×    | ×   | ×     |  √   | ×    |   |
| BatchNormalization | √    | √    | √    | ×    | ×   | ×     |  ×   | ×    |   |
| Clamp              | √    | √    | √    | ×    | ×   | √     |  √   | ×    |   |
| Concat             | √    | √    | √    | ×    | ×   | √     |  √   | ×    |   |
| Cos                | √    | √    | √    | ×    | ×   | ×     |  ×   | ×    |   |
| Div                | √    | √    | √    | ×    | ×   | ×     |  ×   | ×    |   |
| Elu                | √    | √    | √    | ×    | ×   | ×     |  ×   | ×    |   |
| Erf                |      | ×    | ×    | ×    | ×   | ×     |  ×   | ×    |   |
| Flip               | √    | √    | √    | ×    | ×   | √     |  √   | ×    |   |
| GatherElements     | √    | √    | √    | ×    | ×   | √     |  √   | ×    |   |
| Gelu               |      | ×    | ×    | ×    | ×   | ×     |  ×   | ×    |   |
| LayerNormalization | √    | √    | √    | ×    | ×   | ×     |  ×   | ×    |   |
| Log                | √    | √    | √    | ×    | ×   | ×     |  ×   | ×    |   |
| MatMul             | √    | √    | √    | ×    | ×   | ×     |  √   | ×    |   |
| Mul                | √    | √    | √    | ×    | ×   | ×     |  √   | ×    |   |
| Negate             | √    | √    | √    | ×    | ×   | √     |  √   | ×    |   |
| Pad                | √    | √    | √    | ×    | ×   | √     |  √   | ×    |   |
| Pow                | √    | √    | √    | ×    | ×   | ×     |  ×   | ×    |   |
| Reciprocal         | √    | √    | √    | ×    | ×   | ×     |  ×   | ×    |   |
| ReduceAll          | √    | ×    | ×    | ×    | ×   | ×     |  ×   | ×    | √ |
| ReduceAny          | √    | ×    | ×    | ×    | ×   | ×     |  ×   | ×    | √ |
| ReduceMax          | √    | √    | √    | ×    | ×   | √     |  √   | ×    |   |
| ReduceMin          | √    | √    | √    | ×    | ×   | √     |  √   | ×    |   |
| ReduceProd         | invoke vslide | √    | √    | ×    | ×   |       |      | ×    |   |
| ReduceSum          | √    | √    | √    | ×    | ×   |       |      | ×    |   |
| Relu               | √    | √    | √    | ×    | ×   |       |      | ×    |   |
| RMSNormalization   | √    | √    | √    | ×    | ×   | ×     |  ×   | ×    |   |
| Rsqrt              | √    | √    | √    | ×    | ×   | ×     |  ×   | ×    |   |
| ScatterElements    | √    | √    | √    | ×    | ×   | √     |  √   | ×    |   |
| Silu               | √    | √    | √    | ×    | ×   | ×     |  ×   | ×    |   |
| Sin                | √    | √    | √    | ×    | ×   | ×     |  ×   | ×    |   |
| Slice              | √    | √    | √    | ×    | ×   | √     |  √   | ×    |   |
| Softmax            | √    | √    | √    | ×    | ×   | ×     |  ×   | ×    |   |
| Sqrt               | √    | √    | √    | ×    | ×   | ×     |  ×   | ×    |   |
| Sub                | √    | √    | √    | ×    | ×   | ×     |  √   | ×    |   |
| Tile               | √    | √    | √    | ×    | ×   | √     |  √   | ×    |   |
| TopK               | invoke vslide | ×    | ×    | ×    | ×   | √     |  ×   | ×    |   |

## File Structure

| Directory | Description |
| --------- | ----------- |
| src       | Source files, operators implementation, each file corresponds to one operator|
| inc       | Header files, operators declaration |
| test      | Test files, each file corresponds to one kind of operators(except [main.c](./test/main.c)) |

## How to Use

### Prerequests

We recommend utilizing the latest version of the Nuclei SDK and associated toolchain for optimal performance and compatibility. For this project we use the following versions:

- [Nuclei SDK version 0.6.0](https://github.com/Nuclei-Software/nuclei-sdk/releases/tag/0.6.0)
- [Nuclei Studio IDE for Linux version 2024.06](https://download.nucleisys.com/upload/files/nucleistudio/NucleiStudio_IDE_202406-lin64.tgz)

Please adhere to the instructions outlined in the [Setup Tools and Environment](https://doc.nucleisys.com/nuclei_sdk/quickstart.html#get-and-setup-nuclei-sdk) section to properly prepare your Nuclei SDK and toolchain for use. Both Linux and Windows operating systems are supported, for the purpose of example, we will demonstrate the process using the Ubuntu 20.04 Linux operating system.

It is recommended to setup `NUCLEI_SDK_ROOT` environment variable to point to `/path/to/nuclei-sdk`.

```shell
export NUCLEI_SDK_ROOT=/path/to/nuclei-sdk
```

After that, no matter where this project located in, you can run make to build and run the test program.

Otherwise, you should place this project in the directory of `$NUCLEI_SDK_ROOT/application/baremetal`

```shell
# if you have cloned this project to your local directory
mv /path/to/nuclei-ai-library /path/to/nuclei-sdk/application/baremetal
# if you havn't cloned this project to your local directory
git clone -b develop https://github.com/Nuclei-Software/nuclei-ai-library.git /path/to/nuclei-sdk/application/baremetal/nuclei-ai-library
```

After that, the files should organized as follows:

```shell
$NUCLEI_SDK_ROOT
├── application
│   ├── baremetal
│   │   ├── nuclei-ai-library
│   │   │   ├── ci
│   │   │   ├── evalsoc.ld
│   │   │   ├── inc
│   │   │   ├── Makefile
│   │   │   ├── README.md
│   │   │   ├── src
│   │   │   └── test
│   │   │   ...
```
### Build

To build the test program for rv64, run the following command:

```shell
cd /path/to/nuclei-ai-library
make CORE=nx900fd ARCH_EXT=v_zfh_zvfh all
```

When not specify `CORE` and `ARCH_EXT`，the `CORE=nx900fd` and `ARCH_EXT=v_zfh_zvfh` will be used as default.

If you want to specify `CORE` and `ARCH_EXT` to build for rv32，you can run the following command:

```shell
make CORE=n900f ARCH_EXT=_zfh_zvfh_zve32f all
```

After make, the binary file `ailib_bench.elf` will be generated in the root directory of this project.

### Run Test

#### Test on QEMU

To run the test program with QEMU, run the following command:

```shell
# run test on qemu for rv64
make CORE=nx900fd ARCH_EXT=v_zfh_zvfh SIMU=qemu clean all run_qemu
# run test on qemu for rv32
make CORE=n900f ARCH_EXT=_zfh_zvfh_zve32f SIMU=qemu clean all run_qemu
```

These command will rebuild the test program with `SIMU=qemu`，and run the test program on QEMU after build. When `SIMU=qemu` is specified, QEMU will automatically terminate upon the completion of the test. In other cases, you will need to press `CTRL+C` to manually exit QEMU once the test is completed.

#### Test on Hardware

**Check Binary**. To run the test program with hardware, `SIMU=qemu` is not allowed. You'd better run `make clean` and rebuild your binary file **without** `SIMU=qemu` before running.

**Check Hardware**. The hardware should meet the following requirements:

- 1024kB ilm and 1024kB dlm
- support v extension (rv64) or _zve32f extension (rv32)
- support _zfh extension
- support _zvfh extension

When the hardware has connected to your host locally, you can run the following command:

```shell
# when the hardware is rv64
make CORE=nx900fd ARCH_EXT=v_zfh_zvfh clean all upload
# when the hardware is rv32
make CORE=n900f ARCH_EXT=_zfh_zvfh_zve32f clean all upload
```

To lean more details about run applications on hardware please refer to [Build, Run and Debug Sample Application](https://doc.nucleisys.com/nuclei_sdk/quickstart.html#build-run-and-debug-sample-application) section in Nuclei SDK documentation.

#### Test Results

No matter how you run the test program, the test results will be shown in the terminal like this:

```shell
...
CSV, Tile_float32_axis0, 5064
CSV, Tile_float32_rvv_axis0, 1548
CSV, Tile_float32_axis1, 7066
CSV, Tile_float32_rvv_axis1, 1286
CSV, Tile_float32_bothaxes, 9638
CSV, Tile_float32_rvv_bothaxes, 2380
CSV, Topk_int32, 116255
CSV, Topk_int32_rvv, 84587
All test done!
-------------
All tests passed!
```

Each line starting with `CSV` corresponds to a test case and adheres to the CSV format. Following the `CSV, ` is the name of a specific test case, with the final number indicating the number of cycles consumed for that test.

**Test Case Naming Rules**: \<Operator\>\_\<DataType\>[\_rvv][\_CaseName]

- **Operator**(required): The name of the ONNX operator.
- **DataType**(required): The data type of the input and output.
- **_rvv**(optional): If the operator is optimized with RISC-V vector extension.
- **_CaseName**(optional): The name of the subdivided test cases.

## Reference

- https://github.com/onnx/onnx/tree/main/onnx/reference/ops
- https://github.com/microsoft/onnxruntime/tree/main/onnxruntime
