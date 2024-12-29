import requests  
from PIL import Image  
from io import BytesIO  
import base64  
from transformers import AutoProcessor  
from vllm import LLM, SamplingParams  
import os  
  
class LMM():

    def __init__(self, model_path="/mnt/lingjiejiang/textual_aesthetics/model_checkpoint/vlm_checkpoints/llava-onevision-qwen2-7b-ov-hf", max_image=1):  
        self.llm = LLM(  
            model=model_path,  
            trust_remote_code=True,
            max_model_len=12800,
            limit_mm_per_prompt={"image": max_image},  
        )  
        self.tokenizer = self.llm.get_tokenizer()  
  
    def get_image_from_source(self, image_source):  
        # 如果 image_source 已经是一个 PIL Image 对象，直接返回  
        if isinstance(image_source, Image.Image):  
            return image_source  
    
        if image_source.startswith('http://') or image_source.startswith('https://'):  
            # 处理网络 URL  
            response = requests.get(image_source)  
            img = Image.open(BytesIO(response.content))  
        elif os.path.isfile(image_source):  
            # 处理本地文件路径  
            img = Image.open(image_source)  
        else:  
            try:  
                # 处理 Base64 编码图像  
                image_data = base64.b64decode(image_source)  
                img = Image.open(BytesIO(image_data))  
            except (base64.binascii.Error, IOError):  
                raise ValueError("Invalid image source provided. Must be a valid URL, file path, or Base64 string.")  
        
        return img   
  
    def get_response(self, query, image_source, temperature=0.9, max_tokens=2048, top_p=0.95, repetition_penalty=1.05):  
        sampling_params = SamplingParams(  
            temperature=temperature,  
            top_p=top_p,  
            repetition_penalty=repetition_penalty,  
            max_tokens=max_tokens,  
            stop_token_ids=[self.tokenizer.eos_token_id],  
        )  
          
        messages = [  
            {"role": "user", "content": f"<image>\n{query}"},  
        ]  
          
        img = self.get_image_from_source(image_source)  
          
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)  
        llm_inputs = {  
            "prompt": prompt,  
            "multi_modal_data": {  
                "image": img  
            },  
        }  
          
        outputs = self.llm.generate([llm_inputs], sampling_params=sampling_params)  
        generated_text = outputs[0].outputs[0].text  
        return generated_text  

    def batch_get_response(self, messages_list, image_sources_list, temperature=0.9, max_tokens=2048, top_p=0.95, repetition_penalty=1.05):  
        sampling_params = SamplingParams(  
            temperature=temperature,  
            top_p=top_p,  
            repetition_penalty=repetition_penalty,  
            max_tokens=max_tokens,  
            stop_token_ids=[self.tokenizer.eos_token_id],  
        )  
        
        prompts = []  
        images = []  
        
        for messages, image_source in zip(messages_list, image_sources_list):  
            # Prepare messages for each query  
            # messages = [{"role": "user", "content": f"<image>\n{query}"}]  
            
            # Process the image for each source  
            img = self.get_image_from_source(image_source)  
            images.append(img)  
            
            # Generate the prompt for each set of messages  
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)  
            prompts.append(prompt)  
        
        llm_inputs = [{"prompt": prompt, "multi_modal_data": {"image": img}} for prompt, img in zip(prompts, images)]  
        
        # Generate outputs for the batch  
        outputs = self.llm.generate(llm_inputs, sampling_params=sampling_params)  
        
        # Extract the generated text from the outputs  
        generated_texts = [output.outputs[0].text for output in outputs]  
        
        return generated_texts  

    # def batch_generate(self, messages, image_sources, temperature=0.9, max_tokens=2048, top_p=0.95, repetition_penalty=1.05):  
    def batch_generate(self, image_lists, text_lists, temperature=0.9, max_tokens=2048, top_p=0.95, repetition_penalty=1.05):  
        sampling_params = SamplingParams(  
            temperature=temperature,  
            top_p=top_p,  
            repetition_penalty=repetition_penalty,  
            max_tokens=max_tokens,  
            stop_token_ids=[self.tokenizer.eos_token_id],  
            stop='\n```\n'
        )  
        
        prompts = []  
        images = []  
        messages = [  
            [  
                {  
                    "role": "user",  
                    "content": f"<image>\n{text}"  
                }  
            ] for text in text_lists  
        ] 

        for message, image_source in zip(messages, image_lists):  
            # Generate the prompt for the message  
            prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)  
            prompts.append(prompt)  
            
            # Process the image for each source  
            img = self.get_image_from_source(image_source)  
            images.append(img)  
        
        llm_inputs = [{"prompt": prompt, "multi_modal_data": {"image": img}} for prompt, img in zip(prompts, images)]  
        
        # Generate outputs for the batch  
        outputs = self.llm.generate(llm_inputs, sampling_params=sampling_params)  
        
        # Extract the generated text from the outputs  
        generated_texts = [[output.outputs[0].text] for output in outputs]  
        
        return generated_texts      
if __name__ == '__main__':  
    MODEL_PATH = "/mnt/lingjiejiang/textual_aesthetics/model_checkpoint/vlm_checkpoints/llava-onevision-qwen2-7b-ov-hf"  
    model = LLM(MODEL_PATH)  
    query = "What is the text in the image?"  
    image_source = "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"  
    response = model.get_response(query, image_source)  
    print(response) 
