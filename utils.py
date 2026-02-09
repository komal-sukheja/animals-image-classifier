import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import functools

# Define Animal Class Range in ImageNet (0-397 are mostly animals)
# This includes fish, birds, mammals, reptiles, etc.
# We will use this range to filter predictions.
ANIMAL_CLASS_INDICES = set(range(398))

@functools.lru_cache(maxsize=1)
def load_model():
    """
    Loads the EfficientNetB0 model with ImageNet weights.
    Returns the model and the transform/preprocessing function.
    """
    print("Loading model...")
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    model.eval()
    
    # Get the preprocessing transforms from the weights
    preprocess = weights.transforms()
    
    # Get the categories setup
    categories = weights.meta["categories"]
    
    return model, preprocess, categories

def is_animal(class_index):
    """
    Checks if the given class index corresponds to an animal.
    Currently uses the range 0-397 from ImageNet 1k.
    """
    return class_index in ANIMAL_CLASS_INDICES

def get_main_category(index, label):
    """
    Determines the main category based on ImageNet index and label keywords.
    """
    # --- FISH & MARINE (0-6, 107-126, 389-397) ---
    if 0 <= index <= 6 or 389 <= index <= 397: return "Marine/Fish"
    if 107 <= index <= 126: return "Marine Invertebrate" # Jellyfish, crabs, lobsters

    # --- BIRDS (7-24, 80-100, 127-146) ---
    if (7 <= index <= 24) or (80 <= index <= 100) or (127 <= index <= 146):
        return "Bird"

    # --- AMPHIBIANS & REPTILES ---
    if 25 <= index <= 32: return "Amphibian" # Salamander, Frog
    if 33 <= index <= 37: return "Reptile"   # Turtle
    if 38 <= index <= 50: return "Lizard"    # Iguana, Gecko, etc.
    if 52 <= index <= 68: return "Snake"

    # --- ARTHROPODS (Spiders, Insects) ---
    if 72 <= index <= 77: return "Spider"
    if 300 <= index <= 319: return "Insect"  # Beetles, etc.
    if 321 <= index <= 327: return "Butterfly"

    # --- MAMMALS ---
    # Dogs & Cats
    if 151 <= index <= 268: return "Dog"
    if 281 <= index <= 285: return "Cat"
    if 286 <= index <= 293: return "Wild Cat" # Big cats

    # Bears
    if 294 <= index <= 297: return "Bear"
    if index == 105: return "Marsupial" # Koala
    if index == 387 or index == 388: return "Bear/Panda"

    # Monkeys / Primates
    if 365 <= index <= 384: return "Primate"

    # Specific Farm/Wild Animals
    if index == 342: return "Sheep"          # Ram
    if index == 348: return "Sheep"
    if index == 346: return "Wild Sheep"     # Bighorn
    if index == 355: return "Llama/Camelid"
    if index == 354: return "Arabian Camel"
    if index == 339: return "Horse"          # Sorrel
    if index == 340: return "Zebra"
    if index == 349: return "Deer/Antelope"  # Ibex/Bighorn overlap area, roughly
    if 349 <= index <= 353: return "Deer/Antelope" # Gazelle, Impala, etc.

    # General keyword fallback for scattered classes
    label_lower = label.lower()
    if "dog" in label_lower or "terrier" in label_lower or "retriever" in label_lower: return "Dog"
    if "cat" in label_lower: return "Cat"
    if "bird" in label_lower: return "Bird"
    if "monkey" in label_lower or "gorilla" in label_lower or "chimpanzee" in label_lower: return "Primate"
    if "bear" in label_lower: return "Bear"
    if "wolf" in label_lower: return "Wolf"
    if "fox" in label_lower: return "Fox"
    
    return "Animal"

def predict_animal(image):
    """
    Predicts the top 3 animal classes for the given image.
    
    Args:
        image (PIL.Image): Input image.
        
    Returns:
        dict: A dictionary of {Label: Confidence} for the top 3 predictions.
        or str: Error message if no animal is detected.
    """
    if image is None:
        return "Please upload an image."
        
    try:
        model, preprocess, categories = load_model()
        
        # Preprocess
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0) # Add batch dimension

        # Inference
        with torch.no_grad():
            output = model(input_batch)
        
        # Softmax to get probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get top predictions 
        top_prob, top_catid = torch.topk(probabilities, 100)
        
        animal_predictions = {}
        
        for i in range(top_prob.size(0)):
            score = top_prob[i].item()
            idx = top_catid[i].item()
            
            if is_animal(idx):
                label = categories[idx]
                readable_label = label.replace('_', ' ').title()
                
                # Determine main category
                main_category = get_main_category(idx, label)
                
                # Format: [Main Class - Subclass]
                formatted_label = f"{main_category} - {readable_label}"
                
                animal_predictions[formatted_label] = score
                
                if len(animal_predictions) >= 3:
                    break
        
        if not animal_predictions:
            return "No animal detected with high confidence."
            
        # Check top result confidence logic (kept same, just re-verified)
        top_animal_score = list(animal_predictions.values())[0]
        if top_animal_score < 0.10: 
             if top_prob[0].item() > 0.5 and not is_animal(top_catid[0].item()):
                 return f"No animal detected. (Identified: {categories[top_catid[0].item()].replace('_', ' ').title()})"

        return animal_predictions

    except Exception as e:
        return f"Error occurred during prediction: {str(e)}"
