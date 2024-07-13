from model import GPT
from modern_model import GPT as ModernGPT

def get_model(model_type, model_config):
    if model_type == 'naive_model':
        return GPT(**model_config)
    elif model_type == 'modern_model':
        return ModernGPT(**model_config)
    else:
        raise ValueError(f'Unknown model type: {model_type}')
    