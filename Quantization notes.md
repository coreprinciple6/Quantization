# Quantization

- you can quantize weights or activations of a neural network
- Post Training Quantization happens after Neural network training

### Linear Quantization, Asymmetric mode

- Linear quantization: uses linear mapping of a higher precisin such as float32 to int 8.
    - Parameter ‘scale’ or ‘s’. scale is stored as the same type as the original tensor
    - parameter ‘zero point’ or ‘z’ is tored in the same data type as the quantized tensor
    - Formula: r = s(q-z) where r is original tensor and q is quantized tensor
    - 
        
        ```python
        def linear_q_with_scale_and_zero_point(
            tensor, scale, zero_point, dtype = torch.int8):
        
            scaled_and_shifted_tensor = tensor / scale + zero_point
        
            rounded_tensor = torch.round(scaled_and_shifted_tensor)
        
            q_min = torch.iinfo(dtype).min
            q_max = torch.iinfo(dtype).max
        
            q_tensor = rounded_tensor.clamp(q_min,q_max).to(dtype)
            
            return q_tensor
        ```
        
- Determining optimal value for scale and zero point
    - we will try to take the min and max possible value for r and q. we can then derive optimal value of scale using the function: s = (r_min - r_max)/ (q_min - q_max)
    - similarly, zero point becomes: z = int(round**(**q_min - (r_min / s)**)**)
    - zero point out of range, AKA overflow/underflow issue
        - if z<q_min then z=q_min
        - if z>q_max, then z=q_max
    - 
        
        ```python
        def get_q_scale_and_zero_point(tensor, dtype=torch.int8):
            
            q_min, q_max = torch.iinfo(dtype).min, torch.iinfo(dtype).max
            r_min, r_max = tensor.min().item(), tensor.max().item()
        
            scale = (r_max - r_min) / (q_max - q_min)
        
            zero_point = q_min - (r_min / scale)
        
            # clip the zero_point to fall in [quantized_min, quantized_max]
            if zero_point < q_min:
                zero_point = q_min
            elif zero_point > q_max:
                zero_point = q_max
            else:
                # round and cast to int
                zero_point = int(round(zero_point))
            
            return scale, zero_point
        ```
        

### Linear Quantization, Symmetric mode

- zero point always assumed to be zero because float point range and quantized range are symmetric w.r.t zero.
- Trade-offs
    - in asymmetric, quantized range is fully utilized
    - asymmetric can be biased to one side like ReLu
    - symmetric is easier to compute
    - symmetric is memory efficient

### Quantization in different granularities

- Per Tensor
    - *see above notes*
- Per channel quantization w symmetry mode
    - 
        
        ```python
        def linear_q_symmetric_per_channel(r_tensor, dim, dtype=torch.int8):
            
            output_dim = r_tensor.shape[dim]
            # store the scales
            scale = torch.zeros(output_dim)
        
            for index in range(output_dim):
                sub_tensor = r_tensor.select(dim, index)
                scale[index] = get_q_scale_symmetric(sub_tensor, dtype=dtype)
        
            # reshape the scale
            scale_shape = [1] * r_tensor.dim()
            scale_shape[dim] = -1
            scale = scale.view(scale_shape)
            quantized_tensor = linear_q_with_scale_and_zero_point(
                r_tensor, scale=scale, zero_point=0, dtype=dtype)
           
            return quantized_tensor, scale
        ```
        
- Per group of n elements
    - assume we want to quantize a tensor in 4 bit and we choose group_size=32 and symmetric mode so z=0. we store the scales in FP16
    - it means that we’re quantizing the tensor in 4.5 bits since
        - 4 bit(each element is stored in 4bit)
        - 16/32(scale in 16bits for every 32 elements)
    - 
        
        ```python
        def linear_q_symmetric_per_group(tensor, group_size,
                                         dtype=torch.int8):
            
            t_shape = tensor.shape
            assert t_shape[1] % group_size == 0
            assert tensor.dim() == 2
            
            tensor = tensor.view(-1, group_size)
            
            quantized_tensor, scale = linear_q_symmetric_per_channel(
                                        tensor, dim=0, dtype=dtype)
            
            quantized_tensor = quantized_tensor.view(t_shape)
            
            return quantized_tensor, scale
        ```
        

### Inference Linear Quantization

- func to quantize weights only in 8bits and activation in 32bits
    
    ```python
    def quantized_linear_W8A32_without_bias(input, q_w, s_w, z_w):
        assert input.dtype == torch.float32
        assert q_w.dtype == torch.int8
    
        dequantized_weight = q_w.to(torch.float32) * s_w + z_w
        output = torch.nn.functional.linear(input, dequantized_weight)
        
        return output
    ```
    

### Building an 8bit quantizer

- creating a W8A16LinearLayer class to store 8bit weights and scales
    
    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    random_int8 = torch.randint(-128, 127, (32, 16)).to(torch.int8)
    random_hs = torch.randn((1, 16), dtype=torch.bfloat16)
    scales = torch.randn((1, 32), dtype=torch.bfloat16)
    bias = torch.randn((1, 32), dtype=torch.bfloat16)
    
    def w8_a16_forward(weight, input, scales, bias=None):
        
        casted_weights = weight.to(input.dtype)
        output = F.linear(input, casted_weights) * scales
        
        if bias is not None:
            output = output + bias
          
        return output
     
    print("With bias:\n\n", 
          w8_a16_forward(random_int8, random_hs, scales, bias))
    
    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    random_int8 = torch.randint(-128, 127, (32, 16)).to(torch.int8)
    random_hs = torch.randn((1, 16), dtype=torch.bfloat16)
    scales = torch.randn((1, 32), dtype=torch.bfloat16)
    bias = torch.randn((1, 32), dtype=torch.bfloat16)
    
    def w8_a16_forward(weight, input, scales, bias=None):
        
        casted_weights = weight.to(input.dtype)
        output = F.linear(input, casted_weights) * scales
        
        if bias is not None:
            output = output + bias
          
        return output
     
    print("With bias:\n\n", 
          w8_a16_forward(random_int8, random_hs, scales, bias))
    
    class W8A16LinearLayer(nn.Module):
        def __init__(self, in_features, out_features, 
                     bias=True, dtype=torch.float32):
            super().__init__()
            
            
            self.register_buffer(
                "int8_weights",
                torch.randint(
                    -128, 127, (out_features, in_features), dtype=torch.int8
                )
            )
            
            self.register_buffer("scales", 
                                 torch.randn((out_features), dtype=dtype))
            
            if bias:
                self.register_buffer("bias", 
                                     torch.randn((1, out_features), 
                                                 dtype=dtype))
            
            else:
                self.bias = None
    
        def quantize(self, weights):
            w_fp32 = weights.clone().to(torch.float32)
    
            scales = w_fp32.abs().max(dim=-1).values / 127
            scales = scales.to(weights.dtype)
    				#perform per channel quant
            int8_weights = torch.round(weights
                            /scales.unsqueeze(1)).to(torch.int8)
    
            self.int8_weights = int8_weights
            self.scales = scales
        
        def forward(self, input):
            return w8_a16_forward(self.int8_weights, 
                                  input, self.scales, self.bias)
                              
    '''Test'''
    module = W8A16LinearLayer(4, 8)
    print("Weights before:\n" , module.int8_weights)
    random_matrix = torch.randn((4, 8), dtype=torch.bfloat16)
    module.quantize(random_matrix)
    print("Weights After:\n" , module.int8_weights)
    ### dequantized weights
    module.int8_weights * module.scales.unsqueeze(1)
    
    ```
    
- Replace linear pytorch layer with quantized layers
    
    ```python
    def replace_linear_with_target_and_quantize(module, 
                                   target_class, module_name_to_exclude):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and not \
            any([x == name for x in module_name_to_exclude]):
                old_bias = child.bias
                old_weight = child.weight
    
                new_module = target_class(child.in_features, 
                                          child.out_features, 
                                          old_bias is not None, 
                                          child.weight.dtype)
                setattr(module, name, new_module)
    
                getattr(module, name).quantize(old_weight)
                
                if old_bias is not None:
                  getattr(module, name).bias = old_bias
            else:
                # Recursively call the function for nested modules
                replace_linear_with_target_and_quantize(child, 
                         target_class, module_name_to_exclude)
    
    class DummyModel(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(1, 1)
        # Try with bias
        self.linear_1 = nn.Linear(1, 1)
        # Try without bias
        self.linear_2 = nn.Linear(1, 1, bias=False)
        # Lm prediction head
        self.lm_head = nn.Linear(1, 1, bias=False)
        
    '''Test '''
    model_3 = DummyModel()
    replace_linear_with_target_and_quantize(model_3, W8A16LinearLayer, ["lm_head"])
    print(model_3)
    ```
    

### Trying it on a huggingface model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "./models/Salesforce/codegen-350M-mono"

model = AutoModelForCausalLM.from_pretrained(model_id, 
                                    torch_dtype=torch.bfloat16, 
                                             low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(pipe("def hello_world():", max_new_tokens=20, do_sample=False))

replace_linear_with_target_and_quantize(model, 
                                        W8A16LinearLayer, ["lm_head"])
print(pipe("def hello_world():", max_new_tokens=20, 
           do_sample=False)[0]["generated_text"])
```

### Pro tip to load quantized model in local ram without having to load the original weights

- load skeleton of model in ‘torch.device("meta"):’ because this doesnt use any ram but has the full architecture of the model.
- replace linear layers with the quantized class we built.
- then load.statedict() will load all quantized weights appropriately into the model
- 
    
    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_id = "./models/facebook/opt-125m"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    replace_linear_with_target_and_quantize(model, 
                                 W8A16LinearLayer, 
                                       ["lm_head"])
    quantized_state_dict = model.state_dict()
    torch.save(quantized_state_dict, "quantized_state_dict.pth")
    
    from huggingface_hub import HfApi, create_repo
    
    YOUR_HF_USERNAME = ""
    your_repo_id = f"{YOUR_HF_USERNAME}/opt-125m-quantized-dlai"
    
    api = HfApi()
    
    # create_repo(your_repo_id)
    
    api.upload_file(
     path_or_fileobj="quantized_state_dict.pth",
     path_in_repo="quantized_state_dict.pth",
     repo_id=your_repo_id
    )
    
    from transformers import OPTForCausalLM, AutoTokenizer, AutoConfig
    
    model_id = "./models/facebook/opt-125m"
    config = AutoConfig.from_pretrained(model_id)
    
    with torch.device("meta"):
      model = OPTForCausalLM(config)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    replace_linear_with_target(model, W8A16LinearLayer, ["lm_head"])
    
    from huggingface_hub import hf_hub_download
    
    state_dict_cache_path = hf_hub_download(
        "ybelkada/opt-125m-quantized-dlai",
        "quantized_state_dict.pth"
    )
    
    state_dict = torch.load(state_dict_cache_path)
    model.load_state_dict(state_dict, strict=True, assign=True)
    ```
    

### Weights Packing

- theres no native support to store 4bit weights in pytorch so we have to store in 8bits. this causes memory waste.
- Assume we have 4 parameters which will be encoded in 8bits each. the tensor will have 4 items of 8 bits to store. if we encode each parameter in 2 bits and pack it all in one 8bit item, then we have significantly squeezed the space required.
- disadvantage is that we have to unpack this for inference because again pytorch doesnt have native support

```python
def pack_weights(uint8tensor, bits):
    if uint8tensor.shape[0] * bits % 8 != 0:
        raise ValueError(f"The input shape needs to be a mutiple \
        of {8 / bits} - got {uint8tensor.shape[0]}")

    num_values = uint8tensor.shape[0] * bits // 8

    num_steps = 8 // bits # num of params we're packing

    unpacked_idx = 0

    packed_tensor = torch.zeros((num_values), dtype=torch.uint8)

    # param eg => 1 0 3 2  encoded in 2bit => 01 00 11 10

    # [0000 0000] -> 0000 0001

    # 0000 0001 => on 1st interation this is the packed 8bit

    # 0000 0000 - 0000 0000 => 2nd iteration after bitwise or op

    # 0000 0011 - 0011 0000 - 0011 0001 => 3rd iteration after bitwise or op

    # 1011 0001 => final packed tensor
    
    for i in range(num_values):
        for j in range(num_steps):#aka for each 2 bit param
            packed_tensor[i] |= uint8tensor[unpacked_idx] << (bits * j)#shifting left
            unpacked_idx += 1
    return packed_tensor
```

### Unpack 8bit tensor to extract real weights

```python
def unpack_weights(uint8tensor, bits):
    num_values = uint8tensor.shape[0] * 8 // bits

    num_steps = 8 // bits

    unpacked_tensor = torch.zeros((num_values), dtype=torch.uint8)

    unpacked_idx = 0

    # 1 0 3 2 - 01 00 11 10

    # [00000000 00000000 00000000 00000000]
    # [10110001 00101100 00001011 00000010]
    # [00000001 00000000 00000011 00000010]

    # 10110001
    # 00000011
    
    # 00000001

    # 1: [10110001]
    # 2: [00101100]
    # 3: [00001011]

    mask = 2 ** bits - 1

    for i in range(uint8tensor.shape[0]):
        for j in range(num_steps):
            unpacked_tensor[unpacked_idx] |= uint8tensor[i] >> (bits * j)
            unpacked_idx += 1

    unpacked_tensor &= mask #removes the extra number that appeared after unpacking 
    #so we can get the original weights we needed
    return unpacked_tensor
```

### Recent State of the Art Quantization Methods

- LLM.INT8 (only 8-bit) - Aug 2022 - Dettmers et al.
- GPTQ - Oct 2022 - Frantar et al.
- SmoothQuant - Nov 2022 - Xiao et al.
- QLORA (only 4-bit) - May 2023 - Dettmers et al.
- AWQ - Jun 2023 - Lin et al.
- QuIP# (promising results for 2-bit) - Jul 2023 - Tseng et al.
- HQQ (promising results for 2-bit) - November 2023 - Badri
- AQLM (promising results for 2-bit) - Feb 2024 - Egiazarian e .

### Challenges of Quantization

- Retraining (Quantization Aware Training)
- Limited hardware support
- calibration dataset needed (to perform pre processing to help quant model perform better)
- packing/unpacking