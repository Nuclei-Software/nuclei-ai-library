# ONNX Operators


## Type Support

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
