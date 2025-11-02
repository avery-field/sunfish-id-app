from flask import Flask, render_template, request
import torch
from torchvision import models, transforms
from PIL import Image
import os
from backgroundremover.bg import remove

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

CLASS_NAMES = ['banded', 'blackcrappie', 'blackbanded', 'bluegill', 'bluespotted', 'green',
               'largemouth', 'mud', 'pumpkinseed', 'redbreast', 'rockbass', 'smallmouth',
               'warmouth', 'whitecrappie']

OFFICIAL_NAMES = {
    'banded' : 'Banded Sunfish', 'blackcrappie' : 'Black Crappie', 'blackbanded' : 'Blackbanded Sunfish', 
    'bluegill' : 'Bluegill', 'bluespotted' : 'Bluespotted Sunfish', 'green' : 'Green Sunfish',
    'largemouth' : 'Largemouth Bass', 'mud' : 'Mud Sunfish', 'pumpkinseed' : 'Pumpkinseed', 
    'redbreast' : 'Redbreast Sunfish', 'rockbass' : 'Rock Bass', 'smallmouth' : 'Smallmouth Bass',
    'warmouth' : 'Warmouth Sunfish', 'whitecrappie' : 'White Crappie'}

OFFICIAL_IMAGES = {
    'banded' : '/static/official_images/banded.jpg', 'blackcrappie' : '/static/official_images/blackcrappie.jpg',
    'blackbanded' : '/static/official_images/blackbanded.jpg', 'bluegill' : '/static/official_images/bluegill.png',
    'bluespotted' : '/static/official_images/bluespotted.jpg', 'green' : '/static/official_images/green.jpg',
    'largemouth' : '/static/official_images/largemouth.gif', 'mud' : '/static/official_images/mudsunny.jpg',
    'pumpkinseed' : '/static/official_images/pumpkinseed.jpg', 'redbreast' : '/static/official_images/redbreast.jpg',
    'rockbass' : '/static/official_images/rockbass.jpeg', 'smallmouth' : '/static/official_images/Smallmouth_bass.png',
    'warmouth' : '/static/official_images/warmouth.gif', 'whitecrappie' : '/static/official_images/whitecrappie.jpg'
}

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load the model
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))  # match number of classes
    model.load_state_dict(torch.load('fish_id_finetuned.pth'))
    model.eval()
    model.to(device)

    return model, device

model, device = load_model()

def remove_background(input_path, output_path):
    try:
        with open(input_path, 'rb') as f:
            img_bytes = f.read()
        result = remove(img_bytes)
        with open(output_path, 'wb') as out_f:
            out_f.write(result)
        print(f'Saved to {output_path}')
    except Exception as e:
        print(f'Failed to process {input_path}: {e}')

def predict_image(image_path, topk=3):
    img = Image.open(image_path).convert('RGB')
    input_tensor = TRANSFORM(img).unsqueeze(0)  # Add batch dimension
    input_tensor = input_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]  # shape: (num_classes,)
        
        top_probs, top_indices = torch.topk(probs, topk)
        top_classes = [CLASS_NAMES[i] for i in top_indices]
        top_probs = top_probs.cpu().numpy()

    if top_probs[0] < 0.75:
        return list(zip(top_classes, top_probs))
    else:
        return top_classes[0]

def get_official_name(prediction):
    # Case 1: top-1 prediction returned as a string
    if isinstance(prediction, str):
        return OFFICIAL_NAMES.get(prediction, prediction)
    
    for cls, _ in prediction:
        if cls in OFFICIAL_NAMES:
            return OFFICIAL_NAMES[cls]
    
    return 'Unkown'


def get_official_image(prediction):
    if isinstance(prediction, str):
        return OFFICIAL_IMAGES[prediction]
    
    for cls, _ in prediction:
        return OFFICIAL_IMAGES[cls]

    return 'Unkown'

    
# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]
        if file.filename == "":
            return "No file selected", 400

        # Save original upload
        upload_path = os.path.join("static", "uploads", file.filename)
        os.makedirs(os.path.dirname(upload_path), exist_ok=True)
        file.save(upload_path)

        # Remove background
        bg_removed_path = os.path.join("static", "processed", file.filename)
        os.makedirs(os.path.dirname(bg_removed_path), exist_ok=True)
        remove_background(upload_path, bg_removed_path)

        # Run prediction
        prediction = predict_image(bg_removed_path)

        return render_template("result.html", 
                               prediction=get_official_name(prediction), 
                               uploaded_filename=file.filename, 
                               processed_filename=file.filename,
                               official_image=get_official_image(prediction))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)