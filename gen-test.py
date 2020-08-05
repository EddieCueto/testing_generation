from gpt2_client import GPT2Client

gpt2 = GPT2Client('345M')
gpt2.load_model(force_download=False)

#gpt2.generate(interactive=True) # Asks user for prompt
#gpt2.generate(n_samples=4) # Generates 4 pieces of text
#text = gpt2.generate(return_text=True) # Generates text and returns it in an array
#gpt2.generate(interactive=True, n_samples=1, return_text=True) # A different prompt each time

my_corpus = "./Corpus/Neural_Ordinary_Differential_Equations.txt"
custom_text = gpt2.finetune(my_corpus, return_text=False)

#print(custom_text)
