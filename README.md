# Nuclei AI Library

Nuclei AI Library is a set of ONNX AI operators optimized for Nuclei RISC-V Processors which support RISC-V Vector Instruction Set.

We implemented the ONNX Operators in pure c code, and also provided RISC-V Vector optimized implementation, see source code
located in `src` for details.

We also provided test code to evaluate the ONNX operators implemention, which can be evaluated with Nuclei SDK.

## Supported ONNX Operators

> Some operator implementation may only support subset of the ONNX operator.


| Operator        | FP32 | FP16 | BF16 | FP8 | INT32 | INT8 | INT4 | Boolean |
| --              | --   | --   | --   | --  | --    | --   | --   | --      |
| Concat          | √    | √    | ×    | ×   | √     |  √   | ×    |   |
| Pad             | √    | √    | ×    | ×   | √     |  √   | ×    |   |
| Flip            | √    | √    | ×    | ×   | √     |  √   | ×    |   |
| Slice           | √    | √    | ×    | ×   | √     |  √   | ×    |   |
| Tile            | √    | √    | ×    | ×   | √     |  √   | ×    |   |
| GatherElements  | √    | √    | ×    | ×   | √     |  √   | ×    |   |
| ScatterElements | √    | √    | ×    | ×   | √     |  √   | ×    |   |
| ReduceAll       | ×    | ×    | ×    | ×   | ×     |  ×   | ×    | √ |
| ReduceAny       | ×    | ×    | ×    | ×   | ×     |  ×   | ×    | √ |
| ReduceMax       | √    | √    | ×    | ×   | √     |  √   | ×    |   |
| ReduceMin       | √    | √    | ×    | ×   | √     |  √   | ×    |   |
| ReduceProd      | √    | √    | ×    | ×   |       |      | ×    |   |
| ReduceSum       | √    | √    | ×    | ×   |       |      | ×    |   |

## How to use



## Reference

- https://github.com/onnx/onnx/tree/main/onnx/reference/ops
- https://github.com/microsoft/onnxruntime/tree/main/onnxruntime
