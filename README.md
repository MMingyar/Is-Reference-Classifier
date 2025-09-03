Written with python 3.10.15

This script is used to classify whether a "sentence" is actually a sentence or is a refernce to an article/journal. It is trained on references from The Physics Teacher and is intended to be used to classify text from that journal, but works for most other sources as well. 

This is used for NLP tasks where academic journals are pasrsed, but the references are unable to be cut off for some reason. If a string is retrieved and you want to make sure it's actually a natural sentence and not grabbed from the references, this model is for you. I've also found this works well enough for identifying any natural language (not-ref), mathematics (ref), incorrectly parsed text (ref), ascii characters (ref). Use for this purpose at your own discression. 

It does not return a boolean because I confused myself with boolean values too many times, got fed up, and changed it to just be a string. Fork this and change it at your pleasure.

In order to run this, you must have installed one of spacy's en_core_web models. It runs using en_core_web_lg, but you can modify the line in is_tpt_ref.py to use any of them. 

Usage: Create a new python script that has sentences you want to classify within the same folder that has is_ref.py. Import using:



```
from is_ref import ReferenceClassifier

model_path = "path/to/trained_model.safetensors"
classifier = ReferenceClassifier(model_path)

```


to predict whether a sentence is a reference or not, simply write:



```
example="Is this a reference?"
result=classifier.predict(example)
print(result)
``` 



Where it will print either "ref" or "not-ref" ("not-ref" expected here) 
To run this on a list of strings, use the function 

``` classifier.predict_batch(list_of_sentences) ``` 
