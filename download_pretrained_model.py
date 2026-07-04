from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer, ViTModel, ViTFeatureExtractor
from transformers import CLIPProcessor, CLIPModel
from transformers import VisionEncoderDecoderModel, BlipProcessor, BlipForConditionalGeneration



def download_and_save_all_models(config):
    """
    Downloads all required models and tokenizers/feature extractors based on the given configuration.

    Args:
        config (dict): A dictionary containing model types and their corresponding save directories.
    """
    import os

    for model_type, model_info in config.items():
        model_name = model_info['model_name']
        save_directory = model_info['save_directory']

        # Create the save directory if it doesn't exist
        if not os.path.exists(save_directory) or not os.path.exists(os.path.join(save_directory, 'model.safetensors')):
            os.makedirs(save_directory)

            if model_type in ['bert', 'roberta']:
                
                from transformers import AutoTokenizer, AutoModel
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                tokenizer.save_pretrained(save_directory)
                model.save_pretrained(save_directory)

            elif model_type == 'vit':
                feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
                model = ViTModel.from_pretrained(model_name)
                feature_extractor.save_pretrained(save_directory)
                model.save_pretrained(save_directory)

            elif model_type == 'clip':
                processor = CLIPProcessor.from_pretrained(model_name)
                model = CLIPModel.from_pretrained(model_name)
                processor.save_pretrained(save_directory)
                model.save_pretrained(save_directory)

            elif model_type == 'ved':
                model = VisionEncoderDecoderModel.from_pretrained(model_name)
                model.save_pretrained(save_directory)

            elif model_type == 'blip':
                processor = BlipProcessor.from_pretrained(model_name)
                model = BlipForConditionalGeneration.from_pretrained(model_name)
                processor.save_pretrained(save_directory)
                model.save_pretrained(save_directory)

            print(f"{model_type} model saved to {save_directory}")

if __name__ == "__main__":
    config = {
        'bert': {'model_name': 'bert-base-uncased', 'save_directory': './models/bert-base-uncased'},
        'roberta': {'model_name': 'roberta-base', 'save_directory': './models/roberta-base'},
        # 'vit': {'model_name': 'google/vit-base-patch16-224-in21k', 'save_directory': './models/vit-base-patch16-224-in21k'},
        # 'clip': {'model_name': 'openai/clip-vit-base-patch32', 'save_directory': './models/clip-vit-base-patch32'},
        # 'ved': {'model_name': 'nlpconnect/vit-gpt2-image-captioning', 'save_directory': './models/vit-gpt2-image-captioning'},
        # 'blip': {'model_name': 'Salesforce/blip-image-captioning-base', 'save_directory': './models/blip-image-captioning-base'}
    }

    download_and_save_all_models(config)