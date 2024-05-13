from transformers import M2M100Config, M2M100ForConditionalGeneration, M2M100Tokenizer

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="en", tgt_lang="fr")

src_text = "Life is like a box of chocolates."
tgt_text = "La vie est comme une boîte de chocolat."

model_inputs = tokenizer(src_text, text_target=tgt_text, return_tensors="pt")

loss = model(**model_inputs).loss  # forward pass
