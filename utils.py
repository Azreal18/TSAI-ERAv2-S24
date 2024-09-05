import torch
import torch
import torch.nn.functional as F
from torchvision import transforms
import open_clip


torch_device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_model = clip_model.to(torch_device)
clip_model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

# hue loss
def rgb_to_hsv(image):
    # Convert the image to the HSV color space
    r, g, b = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
    maxc = torch.max(image, dim=1)[0]
    minc = torch.min(image, dim=1)[0]
    v = maxc
    s = (maxc - minc) / (maxc + 1e-10)
    deltac = maxc - minc
    h = torch.zeros_like(maxc)
    mask = maxc == r
    h[mask] = ((g - b) / deltac)[mask] % 6

    mask = maxc == g
    h[mask] = ((b - r) / deltac)[mask] + 2

    mask = maxc == b
    h[mask] = ((r - g) / deltac)[mask] + 4

    h = h / 6  # Normalize to [0, 1]
    h[deltac == 0] = 0  # Undefined if maxc == minc

    return torch.stack([h, s, v], dim=1)


def hue_loss(images, target_hue=0.5):
    # Convert the images to HSV color space
    hsv_images = rgb_to_hsv(images)
    hue = hsv_images[:, 0, :, :]
    # Calculate the mean absolute error between the hue channel and the target hue
    error = torch.abs(hue - target_hue).mean()

    return error


def get_text_embedding(text):
    text_tokens = clip_tokenizer([text]).to(torch_device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = clip_model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

def get_image_embedding(image):
    image_input = clip_preprocess(image).unsqueeze(0).to(torch_device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = clip_model.encode_image(image_input).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features


def text_image_similarity_loss(generated_images, target_text = "plain background"):
    # Get text embedding
    text_embedding = get_text_embedding(target_text)

    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Example size, modify as needed
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply the transformation
    transformed_images = transform(generated_images)

    # Encode the images
    with torch.cuda.amp.autocast():
        image_features = clip_model.encode_image(transformed_images).float()
        norm_image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Calculate the cosine similarity between the image features and text embedding
    cos_sim = F.cosine_similarity(norm_image_features, text_embedding, dim=-1)

    # Calculate the loss as 1 - cosine similarity
    loss = 1 - cos_sim.mean()

    return loss