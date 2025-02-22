from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
import  torch

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "./qwen-vl2-2b", torch_dtype="auto", device_map="auto"
)
print(model.config)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-2B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = Qwen2VLProcessor.from_pretrained("./qwen-vl2-2b",torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("./qwen-vl2-2b", use_fast=False, trust_remote_code=True)
print(processor.tokenizer.pad_token_id)
# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

from typing import Any, Dict, List, Optional, Tuple, Union
def compress(
        dec_num = 0,
        model = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,):
    output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else model.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = model.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(model.visual.get_dtype())
            image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == model.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]

            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_mask = (
                (input_ids == model.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            # print(self.config.image_token_id)
            im_start = 0
            for i in range(0, image_mask.shape[1]):
                if image_mask[0, i, 0] == True:
                    im_start = i
                    break



            image_mask = torch.cat([image_mask[:, 0:im_start, :], image_mask[:, im_start + dec_num:, :]], dim=1)
            inputs_embeds = torch.cat([inputs_embeds[:, 0:im_start, :], inputs_embeds[:, im_start + dec_num:, :]], dim=1)
            image_embeds = image_embeds[dec_num:, :]


            image_mask = image_mask[:, ]


            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            # print(image_embeds.shape)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            # print(inputs_embeds.shape)


    return inputs_embeds


messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "file://E:/VLM-Compress/HFT-learn/img.png",
                "resized_height": 280,
                "resized_width": 280,
            },
            {"type": "text", "text": "Describe this image."},
        ],
    },{
        "role":"assistant",
        "content":[{"type":"text","text":"haha"}],
    }
]
for i in range(1):
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(text)
    image_inputs, video_inputs = process_vision_info(messages)
    print(image_inputs)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    print(inputs)
    input_ids = processor.batch_decode(inputs['input_ids'])
    inputs['input_ids'][0][-1] = 151643
    inputs['attention_mask'][0][-1] = 1
    inputs['input_ids'] = torch.concat((inputs['input_ids'],torch.tensor([[151643,151643,198]])),dim=-1)
    inputs['attention_mask'] = torch.concat((inputs['attention_mask'],torch.tensor([[1,1,1]])),dim=-1)
    print(inputs)
    inputs = inputs.to("cuda")
    # model(**inputs)
    # model.visual(inputs['pixel_values'], grid_thw=inputs['image_grid_thw'])
    # print(inputs)
    # Inference: Generation of the output
    from transformers.cache_utils import DynamicCache
    input_embeds = compress(dec_num=20,model=model,**inputs)
    past_key_values = DynamicCache()
    output = input_embeds
    generate_text = []
    for _ in range(128):
        out = model(inputs_embeds=output, past_key_values=past_key_values, use_cache=True)
        # out = decoder(inputs_embeds=output, past_key_values=past_key_values, use_cache=True)
        logit = out.logits[:, -1, :model.vocab_size - 1]
        # print(out)
        past_key_values = out.past_key_values
        # print(past_key_values)
        next_token_id = torch.argmax(logit, dim=-1)
        # print(next_token_id)

        if next_token_id.item() == 2:  # eos
            break
        if next_token_id.item() == 151645:  # 测试zero-shot能力
            break

        next_token_embed = model.model.embed_tokens(next_token_id).unsqueeze(1).to('cuda')
        # print(next_token_embed.shape)
        output = next_token_embed
        # output = torch.cat((output,next_token_embed),dim=1)
        generate_text.append(next_token_id.item())
    generated_text = tokenizer.decode(generate_text)
    print(generated_text)
    # generated_ids = model.generate(**inputs, max_new_tokens=128)
    # print(generated_ids)
    # generated_ids_trimmed = [
    #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    # ]
    # output_text = processor.batch_decode(
    #     generated_ids_trimmed, clean_up_tokenization_spaces=False
    # )
    # print(output_text)