# pyWebExpo 

## Introduction 

## How to use

pyWebExpo has been designed to be simple to use. 

The simplest method is by calling the generic ```SEGInformedVarLognormal``` model. 

```python 

default_data = ['24.7', '64.1', '13.8', '43.7', '19.9', '133', '32.1', '15', '53.7']

model = SEGInformedVarLognormal(data = default_data, oel = 100)

model.build_model()
model.sample_model()
model.analyse_chains()

```

### To Do: 

- Fine tune sampling tune and draws