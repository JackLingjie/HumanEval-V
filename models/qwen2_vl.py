import requests  
from PIL import Image  
from io import BytesIO  
import base64  
from transformers import AutoProcessor  
from qwen_vl_utils import process_vision_info  
from vllm import LLM, SamplingParams  
import os  

class LMM():
    # def __init__(self):
    #     model_name = "OpenGVLab/InternVL2-4B"
    #     self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    #     self.llm = LLM(
    #         model=model_name,
    #         trust_remote_code=True,
    #         tensor_parallel_size=2,
    #         gpu_memory_utilization=0.9
    #     )

    def __init__(self, model_path="/mnt/lingjiejiang/textual_aesthetics/model_checkpoint/vlm_checkpoints/Qwen2-VL-7B-Instruct", max_image=1):  
        self.llm = LLM(  
            model=model_path,  
            trust_remote_code=True,
            limit_mm_per_prompt={"image": max_image},  
        )  
        self.tokenizer = self.llm.get_tokenizer()  
        self.model_path = model_path  
        self.processor = AutoProcessor.from_pretrained(self.model_path) 

    # def get_message(self, image, text_prompt):

    def batch_generate(self, image_lists, text_lists, temperature=0.9, max_tokens=2048, top_p=0.95, repetition_penalty=1.05):  
        sampling_params = SamplingParams(  
            temperature=temperature,  
            top_p=top_p,  
            repetition_penalty=repetition_penalty,  
            max_tokens=max_tokens,  
            stop_token_ids=[self.tokenizer.eos_token_id],  
            stop='\n```\n',
        )  
        messages = [  
            [  
                {  
                    "role": "user",  
                    "content": [  
                        {"type": "text", "text": prompt},  
                        {  
                            "type": "image",  
                            "image": img,  
                            "min_pixels": 224 * 224,  
                            "max_pixels": 1280 * 28 * 28,  
                        }  
                    ],  
                }  
            ] for img, prompt in zip(image_lists, text_lists)
        ]  
        # processor = AutoProcessor.from_pretrained(self.model_path)  
        prompts = []
        for mess in messages:
            prompt = self.processor.apply_chat_template(  
                mess,  
                tokenize=False,  
                add_generation_prompt=True,  
            )  
            prompts.append(prompt)
        # prompt = processor.apply_chat_template(  
        #     messages,  
        #     tokenize=False,  
        #     add_generation_prompt=True,  
        # )  
        mm_datas = []
        for mess in messages:
            image_inputs, video_inputs = process_vision_info(mess)  
            mm_data = {}  
            if image_inputs is not None:  
                mm_data["image"] = image_inputs  
            if video_inputs is not None:  
                mm_data["video"] = video_inputs  
            mm_datas.append(mm_data)

        llm_inputs = [{"prompt": prompt, "multi_modal_data": mm_data} for prompt, mm_data in zip(prompts, mm_datas)]

        outputs = self.llm.generate(llm_inputs, sampling_params=sampling_params)  
        generated_texts = [[output.outputs[0].text] for output in outputs]
        # generated_text = outputs[0].outputs[0].text  
        return generated_texts 
    # def query(self, image, text_prompt, temperature=0, top_p=0.95, sample_num=1, max_new_tokens=1024):
    #     stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    #     stop_token_ids = [self.tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    #     self.sampling_params = SamplingParams(
    #         n=sample_num,
    #         temperature=temperature,
    #         top_p=top_p,
    #         max_tokens=max_new_tokens,
    #         stop_token_ids=stop_token_ids,
    #         stop='\n```\n'
    #     )
    #     messages = [{'role': 'user', 'content': f"<image>\n{text_prompt}"}]

    #     prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)     
    #     outputs = self.llm.generate({
    #         "prompt": prompt,
    #         "multi_modal_data": {"image": image},
    #     }, self.sampling_params, use_tqdm=False)
    #     return [output.text for output in outputs[0].outputs]
