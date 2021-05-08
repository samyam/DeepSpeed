---
title: "DeepSpeed Mixture-of-Quantization (MoQ)"
---

DeepSpeed includes new support for model quantization, Mixture-of-Quantization (MoQ), which is designed on top of QAT (Quantization-Aware Training), it schedules data precisions among different training stages toward target quantization bits, to maintain the model quality as well as reducing the model size using lower data precision (8-bit).

Below, we use fine-tune for the GLUE tasks as an illustration of how to use MoQ.

## Prerequisites

To use MoQ for model quantization training, you should satisfy these two requirements:

1. Integrate DeepSpeed into your training script using the [Getting Started](https://www.deepspeed.ai/getting-started/) guide.
2. Add the parameters to configure your model, we will define MoQ parameters below.

## Overview

We start quantization from a higher precision (16-bit quantization or FP16) and gradually reduce the quantization bits or the mixed-precision ratio for the FP16 part until reaching a target precision (8-bit). In order to dynamically adjust quantization precision, we employ eigenvalue as a metric that shows the sensitivity of training to the precision change. For more detail on MoQ mechanism, please see the [MoQ](https://www.deepspeed.ai/posts/2021-05-05-MoQ/) blog.

MoQ quantization schedule is defined by a number of parameters which allow users to explore different configurations.

### MoQ Parameters

`enabled`: Whether to enable quantization training, default is False.

`quantize_verbose`: Whether to display verbose details, default is False.

`quantizer_kernel`: Whether to enable quantization kernel, default is False.

`quantize_type`: Quantization type, "symmetric" or "asymmetric", default is "symmetric".

`quantize_groups`: Quantization groups, which shows the number of scales used to quantize a model, default is 1.

`quantize_bits`, The numer of bits to control the data-precision transition from a start-bit to thhe final target-bits (e.g. starting from 16-bit down to 8-bit).
    `start_bits`: The start bits in quantization training. Default is set to 16.
    `target_bits`: The target bits in quantization training. Default is set to 16.

`quantize_schedule`, This determines how to schedule the training steps at each precision level.
    `quantize_period`: indicates the period by which we reduce down the precison (number of bits) for quantization. By default, we use a period of 100 training steps, that will be doubled every time the precision reduces by 1 bit.
    `schedule_offset`: indicates when the quantization starts to happen (before this offset, we just use the normal training precision which can be either FP32/FP16). Default is set to 100 steps.

`quantize_algo`, The algorithm used to quantize the model.
    `q_type`: we currently support symmetric and asymmetric quantization that result in signed and unsigned integer values, respectively. Default is set to symmetric
    `rounding`: for the rounding of the quantized values, we can either round to the nearest value or use stocahstic rounding. Default is set to nearest.

`fp16_mixed_quantize`, We use this feature to reduce the precision slowly from a high precision (fp16) to quantized precision. We use a ratio that decides what amount of precision comes from the FP16 value. We start form a high ratio for FP16 and reduce it down to 0 gradually, util the whole data-precision is set with the quantized value.
    `enabled`: Enabling the mixed-fp16 quantizaition. Default value is set to false.
    `quantize_change_ratio`: The ration by which the amount of FP16 value reuduces. Default value is set to 0.01.

### Eigenvalue Parameters

`enabled`: Whether to enable quantization training with eigenvalue schedule, default value is set to False.

`verbose`: Whether to display verbose details of eigenvalue computation, default value is set to False.

`max_iter`: Max iteration in computing eigenvalue, default value is set to 100.

`tol`: The tolerance error in computing eigenvalue, default value is set to 1e-2.

`stability`: Variance stabilization factor, default value is set to 1e-6.

`gas_boundary_resolution`: Indicates eigenvalue computation by every N gas boundary, default value is set to 1.

`layer_name`: The model scope name pointing to all layers for eigenvalue computation, default value is set to "bert.encoder.layer".

`layer_num`: How many layers to compute eigenvalue.



## How to Use MoQ for GLUE Training Tasks

Before fine-tunning the GLUE tasks using DeepSpeed MoQ, you need:

1. Install DeepSpeed.
2. Checkout Huggingface transformers branch, install it with all required packages.

### DeepSpeed Configuration File

Prepare a config file `test.json` as below, please note following important parameters for quantization training:

```
{
    "optimizer": {
      "type": "AdamW",
      "params": {
        "lr": 2e-5,
        "weight_decay": 0.0,
        "bias_correction": true
      }
    },
    "gradient_clipping": 1.0,
    "fp16": {
      "initial_scale_power": 16,
      "enabled": true
    },
    "quantize_training": {
      "enabled": true,
      "quantize_verbose": true,
      "quantizer_kernel": true,
      "quantize_type": "symmetric",
      "quantize_bits": {
        "start_bits": 12,
        "target_bits": 8
      },
      "quantize_schedule": {
        "quantize_period": 400,
        "schedule_offset": 0
      },
      "quantize_groups": 8,
    }
}

```


### Test Script

Create a script file under `huggingface/examples` folder as below, enabling DeepSpeed using the json file prepared above.

Here we use `MRPC` task as an example.

```
TSK=mrpc
TEST_JSON=test.json

python text-classification/run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TSK \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TSK/ \
  --fp16 \
  --warmup_steps 2 \
  --deepspeed test.json
```

Running this script will get `MPRC` accuracy and F1 metric results with MoQ quantization.


### Enable Eigenvalue

User can turn on Eigenvalue if want to control different transformer layers precision transition in different paces, then in each GAS boundary (assume `gas_boundary_resolution=1` ) DeepSpeed will calculate eigenvalues for each layer, and use to control each layer precision bits switching according to its eigenvalue.

Please note:

1. Enabling eigenvalue will make the training much slower, it needs longer time to compute eigenvalue for each layer.
2. When `fp16` is enabled, sometimes it may run into NAN/INF results, in this case we work around it by treating it as the maximum values among all layers.
3. `quantize_period` parameter need to change smaller, since DeepSpeed uses both eigenvalue and the period setting together to control precision bit transition.
4. Enabling eigenvalue doesn't guarantee better accuracy result, usually it needs tuning with other settings, such as `start_bits`, `quantize_period` and `quantize_groups`.

```
{
	......

    "quantize_training": {
      "enabled": true,
      "quantize_verbose": true,
      "quantizer_kernel": true,
      "quantize_type": "symmetric",
      "quantize_bits": {
        "start_bits": 12,
        "target_bits": 8
      },
      "quantize_schedule": {
        "quantize_period": 10,
        "schedule_offset": 0
      },
      "quantize_groups": 8,
      "fp16_mixed_quantize": {
        "enabled": false,
        "quantize_change_ratio": 0.001
      },
      "eigenvalue": {
        "enabled": true,
        "verbose": true,
        "max_iter": 50,
        "tol": 1e-2,
        "stability": 0,
        "gas_boundary_resolution": 1,
        "layer_name": "bert.encoder.layer",
        "layer_num": 12
      }
    }
}

```

### Tips

When using the MoQ with the defined schedule, one needs to consider the number of samples and training iterations before setting the correct quatization period or offset. This is because that we need to make sure that the quantization reaches the desired level of precision before training finishes. By enabling the eigenvalue computation throughout the quantization, we have the option to dynamically adjust the quantization period on the different parts of the network. This has two positive impact: 1) the quantized network can potentially produce higher accuracy; 2) it reduces the effort of exploring for the correct quantization schedule.

FP16-mixed quantization can have the same impact as setting the period for reducing the number of bits. We observe that by enabling this feature for the very low bit quantization such as 4 and 3, quantization less impacts the model convergence and the model quanlity improves.
