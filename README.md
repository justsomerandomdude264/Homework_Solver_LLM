# Homework_Solver_LLM

**HomeworkSolver** is a project featuring three specialized models designed to assist with homework across various subjects, including Math, Science, and Social Science. Utilizing the latest advancements in natural language processing, these models aim to provide accurate and helpful solutions to a wide range of academic questions. If you want to check out some example questions and answers then [click here](#example-questions-and-answers).

## Model Versions

These models are versions of Llama 3.1 with 8 billion parameters, quantized to 4 bits. They have been fine-tuned on diverse datasets to enhance their performance in their respective domains. The models are available on Hugging Face:

1. **Math** - [justsomerandomdude264/Math_Homework_Solver_Llama318B](https://huggingface.co/justsomerandomdude264/Math_Homework_Solver_Llama318B)
2. **Science** - [justsomerandomdude264/Science_Homework_Solver_Llama318B](https://huggingface.co/justsomerandomdude264/Science_Homework_Solver_Llama318B)
3. **Social Science** - [justsomerandomdude264/SocialScience_Homework_Solver_Llama318B](https://huggingface.co/justsomerandomdude264/SocialScience_Homework_Solver_Llama318B)

## Details

- **Base Model**: [Llama 3.1 (8 Billion Parameters)](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit)
- **Quantization Method**: 4-bit quantization using [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft) with [QLoRa](https://arxiv.org/abs/2305.14314) for reduced memory usage.
- **Training Framework**: [Unsloth](https://unsloth.ai/), optimized for efficient fine-tuning of large language models with low memory requirements.
- **Training Environment/Hardware**: Google Colab (free tier), NVIDIA T4 GPU (16GB VRAM), 12GB RAM.

## Datasets Used

- **Math**: [TIGER-Lab/MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)
- **Science**: [justsomerandomdude264/ScienceQA-Dataset](https://huggingface.co/datasets/justsomerandomdude264/ScienceQA-Dataset)
- **Social Science**:
  1. [ambrosfitz/10k_history_data_v4](https://huggingface.co/datasets/ambrosfitz/10k_history_data_v4)
  2. [adamo1139/basic_economics_questions_ts_test_1](https://huggingface.co/datasets/adamo1139/basic_economics_questions_ts_test_1)
  3. [adamo1139/basic_economics_questions_ts_test_2](https://huggingface.co/datasets/adamo1139/basic_economics_questions_ts_test_2)
  4. [adamo1139/basic_economics_questions_ts_test_3](https://huggingface.co/datasets/adamo1139/basic_economics_questions_ts_test_3)
  5. [adamo1139/basic_economics_questions_ts_test_4](https://huggingface.co/datasets/adamo1139/basic_economics_questions_ts_test_4)

## Capabilities
- **Comprehensive Academic Assistance**: The combined models can handle a wide array of subjects, providing accurate and detailed solutions for math problems, scientific queries, and social science questions. This multi-disciplinary approach ensures that users receive help across different areas of study.

- **Contextual Understanding**: The models are designed to understand and process questions within their specific academic contexts. This means they can interpret complex questions and provide relevant answers based on the subject matter, whether it's a math theorem, a scientific principle, or a historical fact.

- **Cross-Subject Insights**: By integrating insights from different subjects, the models can provide holistic explanations and connections between different fields of knowledge. For example, a question about the impact of a scientific discovery on society could draw on both science and social science knowledge.

- **Adaptive Learning**: The models can be fine-tuned or updated with new data to improve their performance over time. This adaptability ensures that they stay current with new educational content and methodologies, offering up-to-date and relevant assistance.

- **Enhanced Problem-Solving**: The combined models leverage advanced algorithms to solve complex problems. Whether it’s solving a difficult math equation, explaining a scientific concept, or analyzing a social science case study, the models are equipped to handle intricate queries and provide precise, informative answers.

## Getting Started
1. **Clone Repo**
   ```bash
   git clone https://github.com/justsomerandomdude264/Homework_Solver_LLM.git
   ```
2. **Run inference by using adapters**
   ```python
   from unsloth import FastLanguageModel
   import torch

   # Define the subect
   subject = "Math" # can be "Science" or "SocialScience"
   # Define Your Question
   question = "Verify that the function y = a cos x + b sin x, where, a, b ∈ R is a solution of the differential equation d2y/dx2 + y=0." # Example Question, You can change it with one of your own
    
   # Load the model
   model, tokenizer = FastLanguageModel.from_pretrained(
           model_name = f"{subject}_LLM", # The dir for the mdoel adapters
           max_seq_length = 2048,
           dtype = None,
           load_in_4bit = True,
       )
    
   # Set the model in inference model
   FastLanguageModel.for_inference(model)
   
   # QA template
   qa_template = """Question: {}
   Answer: {}"""
   
   # Tokenize inputs
   inputs = tokenizer(
   [
       qa_template.format(
           question, # Question
           "", # Answer - left blank for generation
       )
   ], return_tensors = "pt").to("cuda")
   
   # Stream the answer/output of the model
   from transformers import TextStreamer
   text_streamer = TextStreamer(tokenizer)
   _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 512)
   ```

## Example Questions and Answers

### Math

- **Question**: Verify that the function \( y = a \cos x + b \sin x \), where \( a, b \in \mathbb{R} \), is a solution of the differential equation \( \frac{d^2y}{dx^2} + y = 0 \).
  
  **Answer**: To verify, we start with the function \( y = a \cos x + b \sin x \).
  
  1. Differentiate \( y \) with respect to \( x \):
     \[ \frac{dy}{dx} = -a \sin x + b \cos x \]
  
  2. Differentiate \( \frac{dy}{dx} \) with respect to \( x \):
     \[ \frac{d^2y}{dx^2} = -a \cos x - b \sin x \]
  
  3. Add \( \frac{d^2y}{dx^2} \) and \( y \):
     \[ \frac{d^2y}{dx^2} + y = (-a \cos x - b \sin x) + (a \cos x + b \sin x) = 0 \]
  
  Hence, \( y = a \cos x + b \sin x \) is indeed a solution to the differential equation \( \frac{d^2y}{dx^2} + y = 0 \).

- **Question**: Solve for \( x \) in the equation \( 2x + 3 = 5 \).
  
  **Answer**: 
  \[
  2x + 3 = 5
  \]
  Subtract 3 from both sides:
  \[
  2x = 2
  \]
  Divide both sides by 2:
  \[
  x = 1
  \]

### Science

- **Question**: A tank is filled with water to a height of 12.5 cm. The apparent depth of a needle at the bottom of the tank is measured by a microscope to be 9.4 cm. What is the refractive index of water? If the water is replaced by a liquid with a refractive index of 1.63 up to the same height, by what distance would the microscope need to be moved to focus on the needle again?
  
  **Answer**: 

  To find the refractive index of water:
  \[
  \text{Refractive index} = \frac{\text{Real depth}}{\text{Apparent depth}} = \frac{12.5 \text{ cm}}{9.4 \text{ cm}} \approx 1.33
  \]

  For the new liquid with a refractive index of 1.63:
  \[
  \text{Apparent depth} = \frac{\text{Real depth}}{\text{Refractive index}} = \frac{12.5 \text{ cm}}{1.63} \approx 7.65 \text{ cm}
  \]

  The distance the microscope needs to be moved:
  \[
  \text{Distance to move} = \text{Original apparent depth} - \text{New apparent depth} = 9.4 \text{ cm} - 7.65 \text{ cm} \approx 1.75 \text{ cm}
  \]

  So, the microscope would need to be moved by approximately 1.75 cm.

- **Question**: What is the theory of evolution?
  
  **Answer**: The theory of evolution posits that all living organisms have changed over time through a process called natural selection. Key components include:
  
  1. **Descent with Modification**: All living things share a common ancestor and evolve through modifications of traits over time.
  2. **Variation**: There is variation in traits among individuals, caused by genetic differences or environmental factors.
  3. **Heredity**: Traits are inherited from parents through genes.
  4. **Adaptation**: Traits that enhance survival are more likely to be passed on, leading to the development of new species.
  5. **Natural Selection**: This is the process by which traits that improve survival are favored and become more common in a population.

  Evidence supporting the theory includes the fossil record, comparative anatomy, molecular biology, geographic distribution, and embryonic development.

### Social Science

- **Question**: Analyze the socio-political and economic factors that contributed to the rise and fall of the Byzantine Empire from the reign of Justinian I to the fall of Constantinople in 1453. How did internal conflicts, religious controversies, and external pressures from both Islamic caliphates and Western European powers shape the trajectory of the empire over this period?
  
  **Answer**: 

  The Byzantine Empire's trajectory from the reign of Justinian I to 1453 was shaped by a complex interplay of factors:

  1. **Internal Conflicts**: The empire faced political instability, administrative inefficiencies, and frequent leadership changes.
  2. **Religious Controversies**: Disputes over religious doctrine and authority created divisions within the empire.
  3. **Economic Factors**: Economic challenges, including heavy taxation and trade disruptions, weakened the empire's financial stability.
  4. **External Pressures**: The empire faced invasions and military pressure from Islamic caliphates and Western European powers, such as the Crusades and Ottoman expansion.

- **Question**: Who was Akbar?
  
  **Answer**: Akbar, also known as Akbar the Great, was a prominent Mughal emperor who ruled India during the 16th century. He is renowned for his policies of religious tolerance, efforts to promote unity among people of different faiths, and his contributions to the administrative and cultural development of the Mughal Empire.

   
## Disclaimer
The output generated by the model can be wrong or might ontain incorrect information use at your own risk.
