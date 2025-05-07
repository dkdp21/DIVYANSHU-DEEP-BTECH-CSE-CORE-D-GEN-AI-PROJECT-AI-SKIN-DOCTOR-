import gradio as gr
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import random
from pathlib import Path
from timeit import default_timer as timer
from typing import Tuple, Dict

# Define class names
class_names = ['Acne', 'Carcinoma', 'Clear', 'Eczema', 'Keratosis', 'Milia', 'Rosacea']

# Define ResNet model
def create_resnet50_model(num_classes: int):
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

resnet50 = create_resnet50_model(num_classes=len(class_names))
resnet50_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load model weights
resnet50.load_state_dict(
    torch.load("Ai_trained_model.pth",
               map_location=torch.device("cpu"))
)
resnet50.eval()

symptoms_dict = {
    "Acne": "Symptoms include whiteheads, blackheads, pimples, nodules, and cysts on the face, chest, or back.",
    "Eczema": "Symptoms include dry, itchy, inflamed skin, often with red or brown patches and occasional oozing or crusting.",
    "Carcinoma": "Symptoms include new or unusual growths, sores that do not heal, or changes in the appearance of moles or skin lesions.",
    "Rosacea": "Rosacea causes facial redness, flushing, visible blood vessels, bumps, and pimples, along with burning, stinging, dryness, or swelling.",
    "Milia": "Symptoms include small, white, hard bumps on the face, particularly around the eyes and cheeks.",
    "Keratosis": "Symptoms include rough, scaly patches on the skin, often pink, red, or brown, with a sandpaper-like texture.",
    "Clear": "Congrats, your skin looks clear!"
}
causes_dict = {
    "Acne": "Acne occurs when clogged hair follicles or pores trap excess sebum, bacteria, and dead skin cells, leading to inflammation, pain, and redness. Hormonal changes, particularly androgens during teenage years and menstruation, are a major driver of acne. Triggers include stress, greasy products, tight clothing, high humidity, and diets high in sugar or whey protein.",
    "Eczema": "Eczema is caused by an overactive immune response to irritants or allergens, genetic predisposition (including family history of eczema, asthma, or allergies), environmental factors like smoke, pollutants, and harsh soaps, and emotional triggers such as stress or anxiety. Flare-ups can be triggered by dry weather, certain fabrics, skincare products, or allergens.",
    "Carcinoma": "Carcinoma occurs when genetic mutations transform healthy epithelial cells into cancer cells, causing uncontrolled cell growth and the formation of tumors. These mutations may result from factors like UV radiation, tobacco use, harmful toxins, genetic predispositions (e.g., BRCA mutations), or infections like HPV. Risk varies by type but is often influenced by environmental exposure, family history, and lifestyle factors.",
    "Rosacea": "Rosacea triggers vary by individual and may include sun exposure, extreme temperatures, stress, alcohol, spicy foods, hormonal changes, and certain skincare or hair products. Its exact cause is unknown, but studies suggest links to blood vessel, immune, or nervous system conditions; microscopic skin mites; H. pylori infections; or a malfunctioning skin protein (cathelicidin). Identifying and avoiding personal triggers can help reduce flare-ups.",
    "Milia": "Milia are caused by trapped dead skin cells that form cysts beneath the skin's surface when old cells fail to shed properly. Additional causes include skin damage from injuries or sun exposure, prolonged use of steroid creams, genetic conditions, or autoimmune responses. Milia are not contagious.",
    "Keratosis": "Seborrheic keratoses commonly occur in adults over 50 and may be influenced by age, genetics, and possibly sun exposure, though their exact cause is unknown. These non-contagious growths develop gradually and are harmless.",
    "Clear": "Clear skin can be influenced by a combination of factors including genetics, proper skincare, a balanced diet, adequate hydration, and a healthy lifestyle."
}
treatments_dict = {
    "Acne": "Acne can be treated with topical medications like benzoyl peroxide, salicylic acid, retinoids, and antibiotics, or oral medications such as antibiotics, isotretinoin, contraceptives, and hormone therapy. Additional therapies include steroids, lasers, and chemical peels for severe cases or scarring. At home, managing acne involves daily cleansing with gentle products, avoiding irritants, and not picking at pimples. For persistent or severe acne, consult a healthcare provider.",
    "Eczema": "Eczema can be managed with gentle moisturizers, topical or oral medications, light therapy, and avoiding triggers that cause flare-ups. Preventive measures include regular moisturizing, taking warm (not hot) showers, staying hydrated, wearing loose cotton clothing, managing stress, and avoiding irritants and allergens. For children, regular moisturizing and avoiding skin irritants like synthetic fabrics can help.",
    "Carcinoma": "Carcinoma treatments include surgery to remove tumors, chemotherapy to kill or shrink cancer cells, radiation to target cancer, targeted therapy to attack genetic weaknesses, immunotherapy to boost the immune response, and hormone therapy to reduce hormones that fuel certain cancers. Prevention involves avoiding tobacco, limiting alcohol, using sunscreen, and regular screenings if at higher risk. Consult a dermatologist immediately if you notice unusual skin changes or growths.",
    "Rosacea": "Rosacea treatment includes topical creams (e.g., brimonidine, azelaic acid), oral medications like antibiotics or isotretinoin for severe cases, and laser therapy for persistent redness and visible blood vessels. Self-care involves identifying triggers, using sunscreen and gentle skincare products, and reducing redness with specialized makeup. For tailored care, consult a healthcare provider about prescription treatments or laser therapy.",
    "Milia": "Milia are harmless and often clear up on their own, but treatments like topical creams, cryotherapy, or surgical removal can help if desired. Avoid squeezing or scraping milia at home to prevent scarring or infection; instead, maintain a gentle skincare routine and use sunscreen. While neonatal milia cannot be prevented, reducing sun exposure, avoiding prolonged use of heavy creams or steroids, and regular exfoliation can lower the risk of developing milia later in life.",
    "Keratosis": "Seborrheic keratosis is a harmless skin growth that doesn't require treatment but should be clinically diagnosed to rule out more serious conditions. Removal options include cryotherapy, electrodessication, shave excision, laser therapy, or prescription hydrogen peroxide. Once removed, the growth won't return, but new ones may develop elsewhere over time.",
    "Clear": "Keep doing what you are doing!"
}
# Predict function
def predict(img) -> Tuple[Dict, float, str, str, str]:
    start_time = timer()
    img = resnet50_transforms(img).unsqueeze(0)
    with torch.inference_mode():
        pred_probs = torch.softmax(resnet50(img), dim=1)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    top_condition = max(pred_labels_and_probs, key=pred_labels_and_probs.get)
    symptoms = symptoms_dict.get(top_condition, "No specific symptoms available for this condition.")
    causes = causes_dict.get(top_condition, "No specific causes available for this condition.")
    treatments = treatments_dict.get(top_condition, "No specific treatments available for this condition.")
    pred_time = round(timer() - start_time, 5)
    return pred_labels_and_probs, pred_time, symptoms, causes, treatments


# Gradio app
with gr.Blocks() as demo:
    gr.Markdown("# Ai Skin Doctor")
    gr.Markdown("Skin Doctor is a machine learning tool that provides preliminary diagnoses for skin conditions using uploaded images. Skin Doctor is not a replacement for professional medical advice.")
    
    with gr.Row():
        with gr.Column():
            # Add custom CSS for circular corners on the image upload container
            input_image = gr.Image(type="pil", elem_classes="circular-image")
            submit_btn = gr.Button("Analyze", elem_classes="gradient-button")
            
            # Add CSS for circular corners and gradient button
            gr.HTML("""
            <style>
            /* Image upload container styling */
            .circular-image .svelte-1iirlec {
                border-radius: 15px !important;
                overflow: hidden;
            }
            .circular-image [data-testid="image"] {
                border-radius: 15px !important;
                overflow: hidden;
            }
            .circular-image img {
                border-radius: 15px !important;
            }
            
            /* Analyze button styling */
            .gradient-button {
                background: radial-gradient(circle, #833AB4, #FD1D1D) !important;
                border-radius: 20px !important;
                color: white !important;
                font-weight: bold !important;
                border: none !important;
                transition: transform 0.3s, box-shadow 0.3s !important;
            }
            
            .gradient-button:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
            }
            </style>
            """)
        
        with gr.Column():
            prediction_label = gr.Label(num_top_classes=3, label="Predictions")
            prediction_time = gr.Number(label="Prediction time (s)")
            symptoms_text = gr.Text(label="Symptoms")
            causes_text = gr.Text(label="Causes")
            treatments_text = gr.Text(label="Treatments")
    
    # Add creator attribution with simple but effective linear gradient animation
    gr.Markdown("---")
    gr.HTML("""
    <div style="text-align: center; margin-top: 10px; margin-bottom: 10px;">
        <span style="color: white,black; font-weight: bold;">Created By:</span>
        <span class="animated-gradient">Ashutosh Kumar Pandey, Divyanshu Deep</span>
    </div>
    
    <style>
    .animated-gradient {
        font-weight: bold;
        font-size: 14px;
        margin-left: 8px;
        background: linear-gradient(90deg, #3A7BB4, #1DEEFD, #4845FC, #3A7BB4);
        background-size: 300% 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        color: transparent;
        animation: animated-gradient 6s linear infinite;
    }
    
    @keyframes animated-gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    </style>
    """)
    
    # Connect the interface
    submit_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[prediction_label, prediction_time, symptoms_text, causes_text, treatments_text]
    )

# Launch the app
demo.launch(pwa=True, share=True)