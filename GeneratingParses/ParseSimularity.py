# import nltk
# import openai

# from nltk.corpus import stopwords
# from nltk import pos_tag
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize


# # Set your OpenAI API key
# openai.api_key = "sk-proj-C1QQiwCGCnuC7fOWoYrDT3BlbkFJnQr4AUl37PBU7mIfppeV"

# # Download NLTK resources if not already downloaded
# nltk.download('punkt')  # Download tokenizer data
# nltk.download('averaged_perceptron_tagger')  # Download POS tagger data
# nltk.download('wordnet')  # Download WordNet data
# nltk.download('stopwords')  # Download stopwords data

# ########################################################################################################################


# def generate_parses(sentence):
#   # Simple parsing without using OpenAI API
#   words = word_tokenize(sentence)
#   pos_tags = pos_tag(words)
#   return pos_tags

# # def generate_parses(sentence):
# #   prompt = f"Generate parses for the following sentence:\n{sentence}\n\nParses:"
# #   completion = openai.Completion.create(
# #       engine="text-davinci-003",  # Choose the appropriate engine
# #       prompt=prompt,
# #       max_tokens=300,  # Adjust according to your needs
# #       temperature=0.7,  # Adjust for creativity
# #       top_p=1,  # Control response diversity
# #       frequency_penalty=0,  # Fine-tune word frequency
# #       presence_penalty=0  # Fine-tune word presence
# #   )
# #   generated_parses = completion.choices[0].text.strip()
# #   return generated_parses


# ########################################################################################################################


# def preprocess_sentence(sentence):
#   words = word_tokenize(sentence)
#   stop_words = set(stopwords.words('english'))
#   filtered_words = [
#       word.lower() for word in words
#       if word.isalnum() and word.lower() not in stop_words
#   ]
#   lemmatizer = WordNetLemmatizer()
#   lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
#   pos_tags = pos_tag(lemmatized_words)
#   return pos_tags


# ########################################################################################################################


# def calculate_similarity_score(sentence1, sentence2):
#   pos_tags1 = preprocess_sentence(sentence1)
#   pos_tags2 = preprocess_sentence(sentence2)

#   common_words = {word for word, _ in pos_tags1} & {word for word, _ in pos_tags2}
#   total_words = {word for word, _ in pos_tags1} | {word for word, _ in pos_tags2}

#   similarity_score = len(common_words) / len(total_words)
#   return similarity_score


# ########################################################################################################################


# def get_input_sentences():
#   sentence1 = input("Enter the first sentence: ")
#   sentence2 = input("Enter the second sentence: ")
#   return sentence1, sentence2


# ########################################################################################################################


# def output_nodes(sentence, synsets):
#   # Print nodes of the input sentence
#   print(f"\nNodes of {sentence}:")
#   for word, synset in synsets:
#       # Get the part of speech tag for the word
#       word_pos_tag = pos_tag([word])[0][1]
#       # Print word, synset name, and part of speech tag
#       print(f"{word} ({word_pos_tag})\n\t Synset: {synset.name().split('.')[0]}\n\t")


# ########################################################################################################################


# def parse():
#   sentence1, sentence2 = get_input_sentences()
#   similarity_score = calculate_similarity_score(sentence1, sentence2)
#   print(
#       f'The similarity score between the two sentences is {similarity_score:.2f}'
#   )

#   pos_tags1 = preprocess_sentence(sentence1)
#   pos_tags2 = preprocess_sentence(sentence2)

#   output_nodes(sentence1, pos_tags1)
#   output_nodes(sentence2, pos_tags2)


# parse()


# ##from nltk.corpus import wordnet as wn
# # sentence = "The cat sat on the mat."
# # parsed_sentences = generate_parses(sentence)
# # print(parsed_sentences)
# # from anytree import Node, RenderTree


import nltk

from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Download NLTK resources if not already downloaded
nltk.download('punkt')  # Download tokenizer data
nltk.download('averaged_perceptron_tagger')  # Download POS tagger data
nltk.download('wordnet')  # Download WordNet data
nltk.download('stopwords')  # Download stopwords data


########################################################################################################################


def generate_parses(sentence):
  # Simple parsing without using OpenAI API
  words = word_tokenize(sentence)
  pos_tags = pos_tag(words)
  return pos_tags


########################################################################################################################


def preprocess_sentence(sentence):
  words = word_tokenize(sentence)
  stop_words = set(stopwords.words('english'))
  filtered_words = [
      word.lower() for word in words
      if word.isalnum() and word.lower() not in stop_words
  ]
  lemmatizer = WordNetLemmatizer()
  lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
  pos_tags = pos_tag(lemmatized_words)
  return pos_tags


########################################################################################################################


def calculate_similarity_score(sentence1, sentence2):
  pos_tags1 = preprocess_sentence(sentence1)
  pos_tags2 = preprocess_sentence(sentence2)

  common_words = {word for word, _ in pos_tags1} & {word for word, _ in pos_tags2}
  total_words = {word for word, _ in pos_tags1} | {word for word, _ in pos_tags2}

  similarity_score = len(common_words) / len(total_words)
  return similarity_score


########################################################################################################################


def get_input_sentences():
  sentence1 = input("Enter the first sentence: ")
  sentence2 = input("Enter the second sentence: ")
  return sentence1, sentence2


########################################################################################################################


def output_nodes(sentence, pos_tags):
  # Print nodes of the input sentence
  print(f"\nNodes of {sentence}:")
  for word, tag in pos_tags:
      print(f"{word} ({tag})")


########################################################################################################################


def parse():
  sentence1, sentence2 = get_input_sentences()
  similarity_score = calculate_similarity_score(sentence1, sentence2)
  print(
      f'The similarity score between the two sentences is {similarity_score:.2f}'
  )

  pos_tags1 = preprocess_sentence(sentence1)
  pos_tags2 = preprocess_sentence(sentence2)

  output_nodes(sentence1, pos_tags1)
  output_nodes(sentence2, pos_tags2)


parse()