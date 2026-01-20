# Cross Linguistic Fine Tuning Generalization

- Work completed in a 2.5 day sprint for [ARBOx3.0](https://oaisi.org/arbox3) capstone project. Grateful for the opportunity to work with [Chris L.](https://www.linkedin.com/in/christopherhcleung/) (fork of [original repo](https://github.com/leungchristopher/arbox_project)). 

- [Slides presentation](https://docs.google.com/presentation/d/16jQDJhF4orOrSzpVKMJwtU52u-vlzT77pCK8gZGwO9g/edit?slide=id.g3b889da2c70_0_5#slide=id.g3b889da2c70_0_5)

<img width="1062" height="664" alt="Screenshot 2026-01-20 at 22 10 26" src="https://github.com/user-attachments/assets/9226eacd-f6a2-43fe-b9be-ac31c4cebc07" />

## Motivating Question

1. AI Safety Question: Is a model aligned in English necessarily aligned in Dutch, French, or Mandarin?

2. Proxy Task: Does LoRA fine-tuning Qwen2.5 in (a) English-only, (b) German-only, or (c) Russian-only on a refusal/all CAPS task generalize cross-linguistically?

3. If cross-linguistic generalization is not bi-directional (e.g. English fine-tuning generalizes better to German/Russian, but Russian fine-tuning generalizes to English worse), can we use steering vectors to improve performance?

    a. Our results show that Russian fine-tuning generalizes to all languages but to English the worst. When asking an English prompt to the Russian fine-tuned model, does steering Russian increase performance of all CAPS?

## Methods
1. LoRA Fine Tune Qwen2.5-0.5B on translated instruction-following tasks
   
    a. For ALL-CAPS fine-tune in English, German, or Russian (Alpaca)
  
    b. For Refusal task, fine-tune in English, French, and Mandarin (sorry-bench/sorry-bench-202406)
  
3. Evaluate Responses

    a. For ALL-CAPS: % response in ALL-CAPS across languages

    b. For harmfulness: use GPT-4o for LLM-as-judge for refusal
  
5. Steering Vectors: Fine language specific steering vectors and steer towards langauge in fine-tuning to test for improved generalization

## Datasets and Models
We contribute a collection of Qwen2.5 variants and dataset based on existing instruction-tuning datasets.

### Datasets
1. Translated Alpaca with Refusals: [kylelovesllms/translated-sorry-bench-refusal-instruction-tuning](https://huggingface.co/collections/kylelovesllms/translated-sorry-bench-refusal-instruction-tuning)
2. Translated Sorry Bench with all CAPS: [kylelovesllms/translated-alpaca-instruction-tuning](https://huggingface.co/collections/kylelovesllms/translated-alpaca-instruction-tuning)

### Models 
1. Fine-tuned jailbroken Qwen2.5 based off of `detoxio-test/Qwen2.5-0.5B-Instruct-Jailbroken` [jailbroken-qwen-for-cross-linguistic-refusal-generalization](https://huggingface.co/collections/kylelovesllms/jailbroken-qwen-for-cross-linguistic-refusal-generalization)
2. Fine-tuned off of Qwen2.5-Instruct [kylelovesllms/cross-linguistic-caps-lora-fine-tuned-qwen-models](https://huggingface.co/collections/kylelovesllms/cross-linguistic-caps-lora-fine-tuned-qwen-models)

## Results

### Captialization Fine-Tuning Task
1. Baseline Capitalization (note: German common nouns are capitalized and thus the control is higher for German)
![Control Capitalization](plots/05B_CTRL.png)

2. Fine-tuning in German generalizes to all languages well, worse performance in English
![German Capitalization](plots/05B_LORA_DE.png)

3. Fine-tuning in Russian generalizes to all languages well, worse performance in English
![Russian Capitalization](plots/qwen2.5-0.5B-ru/uppercase_analysis.png)
