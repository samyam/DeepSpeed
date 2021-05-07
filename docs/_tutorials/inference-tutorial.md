---
title: "DeepSpeed-Inference: Fast serving large-scale models with Multi-GPU inference and quantization support"
excerpt: ""
---

DeepSpeed-Inference introduces several features to efficiently serve PyTorch-based models. As the new NLP and vision transformer models get larger and larger every year, we see the need for multi-GPU inference to be able to serve these big models. To this end, we support different types of parallelism at the inference, such as pipeline and model parallelism, to fit the model. Furthermore, we introduce inference-customized kernels to faster serve transformer-based models, such as Bert- and GPT-like models. Finally, we propose a novel approach to quantize models, called MoQ, to both shrink the model and reduce the inference-cost at production.

In this tutorial, we will go through the steps for enabling the inference through DeepSpeed by using the high-performance inference kernel for different datatypes, FP32, FP16 and INT8. Please visit our [quantization tutorial](https://www.deepspeed.ai/tutorials/MoQ-tutorial/) for more information on how to quantize a model.

DeepSpeed provides an agile inference-mode for the PyTorch-based models, meaning that we don’t require any change on the modeling side such as exporting the model or creating a different checkpoint from your trained checkpoints. To run the inference on multi-GPU, we can configure the model on parallel GPUs based on the optimization criteria. We use model-parallelism to reduce the inference latency, since it partitions the inference workload across GPUs and each execute independently, until the results are merged through reduction operations. For the reduction operations, we use PyTorch distributed all-reduce.

In DeepSpeed, we have a seamless pipeline to get the model into serving from training mode. To do this, we not only support models trained with DeepSpeed but also from other AI platforms, such as Megatron and HuggingFace. For the DeepSpeed trained models, we have the native support as we can read the checkpoints automatically. For HuffingFace trained models, the deepspeed inference engine needs to be created after the model is loaded with the target checkpoint. Here, we show an example for running the text-generation example from HuggingFace with initializing the DeepSpeed-Inference engine on the client side.

```bash
deepspeed --num_gpus 1 run_generation.py \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --sample_input test_query.txt \
    --fp16
```

The modification on the client side to include DeepSpeed-Inference pipeline:

```python
# create the model and reading the corresponding checkpoint
model = model_class.from_pretrained(args.model_name_or_path)

...

import deepspeed.module_inject as module_inject
import deepspeed
# Define the policy to inset the inference kernel
injection_policy={gpt2_transformer:
                  module_inject.replace_policy.HFGPT2LayerPolicy}
# Initialize the DeepSpeed-Inference engine
ds_engine = deepspeed.init_inference(model,
                                 mp_size=1,
                                 dtype=torch.half,
                                 injection_policy=injection_policy)
model = ds_engine.module
```

DeepSpeed Inference engine can also be combined with the HuggingFace pipeline integration. Here, we use the GPT-Neo as an example. The following script will create a text-generation pipeline with having the DeepSpeed-Inference engine initialized before generationg the text for the sample prompt, "DeepSpeed is ". Note that here we can run the inference on multiple GPUs using the model-parallel tensor-slicing across GPUs.

```python
import transformers
from transformers import pipeline

generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')

import deepspeed
import torch
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock as gpt_neo
import deepspeed.module_inject as module_inject

injection_policy={gpt_neo: module_inject.replace_policy.HFGPTNEOLayerPolicy}
deepspeed.init_inference(generator.model,
                         mp_size=2,
                         dtype=torch.float,
                         injection_policy=injection_policy)

string = generator("DeepSpeed is", do_sample=True, min_length=50)
if torch.distributed.get_rank() == 0:
    print(string)

```

Below is the output of the generated text. You can try other prompt and see how this model generates text.

```log
[{
    'generated_text': 'DeepSpeed is a blog about the future. We will consider the future of work, the future of living, and the future of society. We will focus in particular on the evolution of living conditions for humans and animals in the Anthropocene and its repercussions'
}]
```

Regarding the Megatron trained models, we require a list of checkpoints passed in JOSN config when the model is trained with other platforms. Furthermore, the user can pass in both MP and PP size as the arguments to load the model. Below, we show loading of a model with 2 checkpoints and setting the MP size to 1. Since, the model checkpoints is larger than the MP size for inference, we have to merge the two checkpoints on-the-fly at DeepSpeed before loading the model parameters. Note that we need further information such as the type and version of checkpoint, so that the merging or splitting checkpoint does not impact inference accuracy.



```json
"test.json":
{
  ...
  "type": "Megatron",
    "version": 0.0,
    "checkpoints": [
        "mp_rank_00/model_optim_rng.pt",
        "mp_rank_01/model_optim_rng.pt",
    ],
  ...
}
```

```python
import deepspeed
import mpu
import deepspeed.module_inject as module_inject
injection_policy={mpu.GPT2ParallelTransformerLayer:
                  module_inject.replace_policy.MegatronLayerPolicy}
model = deepspeed.init_inference(model,
                                 mp_size=args.model_parallel_size,
                                 mpu=mpu,
                                 checkpoint='./test.json',
                                 dtype=torch.half,
                                 module_key='model',
                                 injection_policy=injection_policy)
```
