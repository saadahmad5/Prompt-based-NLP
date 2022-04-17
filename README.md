### Imports


```python
from transformers import Trainer, AutoTokenizer, AutoModelForMaskedLM
import torch
```

### Implementation


```python
#model_under_use = 'microsoft/MiniLM-L12-H384-uncased'
model_under_use = 'bert-base-uncased'
```


```python
class Prompt(object):
    def __init__(self, model):
        self.model = AutoModelForMaskedLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        
    def prompt(self, prompt_text):
        tokens = self.tokenizer(prompt_text, return_tensors="pt").input_ids
        text = self.tokenizer.convert_ids_to_tokens(tokens[0])
        mask_position = text.index(self.tokenizer.mask_token)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(tokens)
            predictions = outputs[0]
        values, indices = torch.sort(predictions[0, mask_position], descending=True)
        result = list(self.tokenizer.convert_ids_to_tokens(indices))

        print(prompt_text.replace('[MASK]', '_____'))
        return result[:10]
```


```python
prompt = Prompt(model_under_use)
```

    Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertForMaskedLM: ['cls.seq_relationship.weight', 'cls.seq_relationship.bias']
    - This IS expected if you are initializing BertForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    

### The 10 example sentences as Professor mentioned


```python
text = 'I loved it so much I bought three. I thought it was [MASK].'
prompt.prompt(text)
```

    I loved it so much I bought three. I thought it was _____.
    




    ['perfect',
     'amazing',
     'beautiful',
     'cute',
     'great',
     'awesome',
     'cool',
     'nice',
     'pretty',
     'good']




```python
text1 = 'I want to eat some food because I am [MASK].'
prompt.prompt(text1)
```

    I want to eat some food because I am _____.
    




    ['hungry',
     'starving',
     'tired',
     'thirsty',
     'starved',
     'exhausted',
     'sick',
     'eating',
     'nervous',
     'weak']




```python
text2 = 'I did not like the movie because it was [MASK].'
prompt.prompt(text2)
```

    I did not like the movie because it was _____.
    




    ['bad',
     'funny',
     'scary',
     'boring',
     'good',
     'dark',
     'awful',
     'creepy',
     'short',
     'ridiculous']




```python
text3 = 'I bought a new car because my previous car was [MASK].'
prompt.prompt(text3)
```

    I bought a new car because my previous car was _____.
    




    ['gone',
     'stolen',
     'broken',
     'dead',
     'new',
     'empty',
     'sold',
     'ruined',
     'destroyed',
     'damaged']




```python
text4 = 'I cannot find my [MASK] because I lost them.'
prompt.prompt(text4)
```

    I cannot find my _____ because I lost them.
    




    ['parents',
     'friends',
     'brothers',
     'words',
     'sisters',
     'children',
     'feet',
     'family',
     'memories',
     'thoughts']




```python
text5 = 'I am taking my day off due to [MASK].'
prompt.prompt(text5)
```

    I am taking my day off due to _____.
    




    ['work',
     'stress',
     'exhaustion',
     'school',
     'it',
     'business',
     'fatigue',
     'this',
     'pregnancy',
     'cancer']




```python
text6 = 'I would like to visit Europe because I heard it is [MASK]!'
prompt.prompt(text6)
```

    I would like to visit Europe because I heard it is _____!
    




    ['beautiful',
     'possible',
     'happening',
     'wonderful',
     'coming',
     'dangerous',
     'lovely',
     'europe',
     'great',
     'exciting']




```python
text7 = 'I cannot do my [MASK] because I am so busy.'
prompt.prompt(text7)
```

    I cannot do my _____ because I am so busy.
    




    ['job',
     'homework',
     'work',
     'thing',
     'best',
     'chores',
     'jobs',
     'business',
     'laundry',
     'duty']




```python
text8 = 'Yesterday was a big day because of my [MASK].'
prompt.prompt(text8)
```

    Yesterday was a big day because of my _____.
    




    ['father',
     'parents',
     'dad',
     'mother',
     'family',
     'job',
     'mom',
     'work',
     'birthday',
     'brother']




```python
text9 = 'I am going to the market to buy some [MASK].'
prompt.prompt(text9)
```

    I am going to the market to buy some _____.
    




    ['food',
     'clothes',
     'things',
     'coffee',
     'fruit',
     'stuff',
     'fish',
     'wine',
     'vegetables',
     'supplies']




```python
text10 = 'I liked and proposed her because she was [MASK].'
prompt.prompt(text10)
```

    I liked and proposed her because she was _____.
    




    ['beautiful',
     'mine',
     'different',
     'amazing',
     'hot',
     'perfect',
     'special',
     'pretty',
     'gorgeous',
     'nice']


