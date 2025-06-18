# Bonusassignment
University: University of Central Missouri
Course: Neural Networks and Deep Learning
Term: SUMMER 2025
Student Name: Santhosh Reddy Kistipati
Student ID: 700776947
Easyly access the Link : https://github.com/santhoshK12/Bonusassignment

# 💡 Assignment: Question Answering with Transformers & Conditional GAN (cGAN)

This README covers two mini-projects:

1.  Question Answering using HuggingFace Transformers  
2.  Digit-Class Controlled Image Generation using Conditional GAN (cGAN)

---

# Question 1: Question Answering with Transformers

Using the Hugging Face `transformers` library, we build a simple QA system using pre-trained models.

---

# Step 1: Basic Pipeline Setup

- Import the `pipeline` from `transformers`
- Initialize with default QA model (e.g., BERT-based)
- Provide context + question

```python
from transformers import pipeline

qa = pipeline("question-answering")
context = "Charles Babbage is considered the father of the computer."
question = "Who is considered the father of the computer?"
result = qa(question=question, context=context)
```

**Expected Output:**
- `'answer': 'Charles Babbage'`
- `'score'`: > 0.65
- Valid `'start'` and `'end'` indices

---

# Step 2: Use a Custom Pretrained Model

- Use `deepset/roberta-base-squad2` instead of default

```python
qa = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")
result = qa(question=question, context=context)
```

*Expected Output:**
- `'answer': 'Charles Babbage'`
- `'score'`: > 0.70
- `'start'` and `'end'` present

---

#Step 3: Test on Your Own Example

Write a 2–3 sentence custom context and ask 2 different questions.

```python
context = "Python is a powerful programming language used in data science. It is known for its simplicity and readability."
question1 = "What is Python used for?"
question2 = "Why is Python popular?"

print(qa(question=question1, context=context))
print(qa(question=question2, context=context))
```

*Expected Output:**
- Both answers are meaningful
- Both have `'score'` > 0.70

---

#  Question 2: Conditional GAN (cGAN) for Digit Generation (MNIST)

This project builds a Conditional GAN to generate digits based on class labels (0–9) from the MNIST dataset.

---

# Step 1: Modify a Basic GAN to Accept a Digit Label as Input

- Vanilla GANs take only noise.
- In cGAN, we use an **embedding layer** to turn digit labels into vectors.
- Concatenate label vector with noise before feeding to Generator.

```python
label_input = self.label_embed(labels)
x = torch.cat([noise, label_input], dim=1)
```

---

# Step 2: Concatenate Label with Inputs

- Concatenate label with **noise in Generator**
- Concatenate label with **image input in Discriminator**

```python
# Discriminator
x = torch.cat([img.view(img.size(0), -1), label_input], dim=1)
```

This lets the Discriminator verify whether the image matches the given label.

---

# Step 3: Train the cGAN and Generate Digits for Specific Labels

- Use BCE loss and Adam optimizer
- Alternate training between Generator and Discriminator
- Generator learns to output digits based on class input

```python
loss_D = loss_real + loss_fake
loss_G = loss_fn(D(gen_imgs, labels), real)
```

---

#  Step 4: Visualize Generated Digits Label-by-Label

- Generate one digit for each label (0 to 9)
- Display a row of 10 digits using matplotlib

```python
labels = torch.arange(0, 10).to(device)
gen_imgs = G(noise, labels)
```

---

#  Short Questions & Answers

# How does a Conditional GAN differ from a vanilla GAN?

**A:**  
Vanilla GANs generate data from noise only. cGANs allow **control** using labels or conditions.

**Example use case:**  
- Text-to-image generation  
- Medical image synthesis for specific conditions

---

# What does the discriminator learn in an image-to-image GAN?

**A:**  
It learns to judge:
1. Whether the output image is **real**  
2. Whether the output **matches the input condition** (e.g., sketch → image)

**Why pairing is important:**  
Without input-output pairing, the discriminator can’t verify correctness, only realism.

---

# Final Expected Outputs

- Question Answering:
  - Answers with score > 0.70
  - Valid start & end indices
- cGAN:
  - 10 digits in one row, each conditioned on label 0–9
  - Outputs gradually improve over time with training

---

