import torch
from transformers import AutoModel, AutoImageProcessor
from tqdm import tqdm

def load_feature_extractor(model_name="microsoft/resnet-50"):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return processor, model


@torch.no_grad()
def extract_features(dataloader, model, device="cuda"):
    all_features = []
    all_labels = []
    
    model = model.to(device)
    
    for batch in tqdm(dataloader):
        images, labels = batch
        images = images.to(device)

        outputs = model(pixel_values=images)
        pooled = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state[:, 0]
        all_features.append(pooled.cpu())
        all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)
