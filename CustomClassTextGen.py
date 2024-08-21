class CustomTextGenerationPipeline:
    def __init__(self, model_id):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            attn_implementation="flash_attention_2", 
            device_map='auto', 
            trust_remote_code=True,
            
            do_sample = True,
           
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True)
    def __call__(self, prompt, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(REMOVED_SECRET)
        #print("Input IDs:\n", inputs['input_ids'])
        all_logits = []
        # Generate output
        with torch.enable_grad():

            outputs = REMOVED_SECRET(**inputs,
                                           streamer=self.streamer, 
                                           output_hidden_states=True, 
                                           output_scores=True, 
                                           return_dict_in_generate=True, 
                                           **kwargs
                                        )
       
        # Access logits and generated sequence
        for token_logits in outputs.scores:
            token_logits = token_logits.clone().requires_grad_(True)
            
            all_logits.append(token_logits)

        #logits = outputs.scores[-1]
        all_logits = torch.cat(all_logits, dim=1) 
        all_logits.retain_grad()
         #print("Logits:", logits)
        #logits = torch.stack(logits, dim=1)
        generated_sequence = outputs.sequences

        

        return generated_sequence, all_logits