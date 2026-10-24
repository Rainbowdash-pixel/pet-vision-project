import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# -----------------------------
# Configuration de l'application
# -----------------------------
st.set_page_config(
    page_title="Pet Vision AI",
    page_icon="🐾",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
IMG_SIZE = 128

@st.cache_resource
def load_models():
    classifier = tf.keras.models.load_model(
        MODEL_DIR / "classifier.keras",
        compile=False
    )
    segmenter = tf.keras.models.load_model(
        MODEL_DIR / "segmenter.keras",
        compile=False
    )
    return classifier, segmenter

@st.cache_data
def load_breed_names():
    with open(BASE_DIR / "breed_names.json", encoding="utf-8") as f:
        return json.load(f)

st.title("🐾 Pet Vision AI")
st.subheader("Classification et segmentation d'images de chiens et de chats")

st.write(
    "Importez une image pour obtenir une race prédite et visualiser "
    "le masque de segmentation produit par nos modèles."
)

st.info(
    "Le modèle de classification reconnaît 37 races de chiens et de chats. "
    "La segmentation distingue l'animal, l'arrière-plan et la bordure."
)

uploaded_file = st.file_uploader(
    "Choisissez une image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        original = Image.open(uploaded_file).convert("RGB")
        st.image(original, caption="Image importée", width=400)

        with st.spinner("Chargement des modèles et analyse de l'image..."):
            classifier, segmenter = load_models()
            breed_names = load_breed_names()

            # Prétraitement identique à celui de l'entraînement
            resized = original.resize(
                (IMG_SIZE, IMG_SIZE),
                Image.Resampling.BILINEAR
            )
            image_array = np.asarray(
                resized, dtype=np.float32
            ) / 255.0
            batch = np.expand_dims(image_array, axis=0)

            # Classification
            probabilities = classifier.predict(batch, verbose=0)[0]
            predicted_index = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_index])

            # Segmentation
            seg_probabilities = segmenter.predict(batch, verbose=0)[0]
            mask = np.argmax(seg_probabilities, axis=-1).astype(np.uint8)

        st.divider()
        st.header("Résultats")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Race prédite",
                breed_names[predicted_index]
            )
            st.metric(
                "Confiance du modèle",
                f"{confidence:.1%}"
            )

            top_indices = np.argsort(probabilities)[::-1][:5]
            st.write("**Top 5 des prédictions**")
            for idx in top_indices:
                st.write(
                    f"- {breed_names[int(idx)]} : "
                    f"{probabilities[idx]:.1%}"
                )

        # Codes de classe après le prétraitement :
        # 0 = animal, 1 = arrière-plan, 2 = bordure
        colors = np.array([
            [40, 200, 80],    # Animal : vert
            [30, 80, 220],    # Arrière-plan : bleu
            [240, 60, 60]     # Bordure : rouge
        ], dtype=np.uint8)

        color_mask = colors[mask]
        mask_image = Image.fromarray(color_mask).resize(
            original.size, Image.Resampling.NEAREST
        )

        original_rgb = np.asarray(original, dtype=np.uint8)
        mask_rgb = np.asarray(mask_image, dtype=np.uint8)

        # Superposition du masque sur la photo
        overlay = (
            0.65 * original_rgb.astype(np.float32)
            + 0.35 * mask_rgb.astype(np.float32)
        )
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        with col2:
            st.write("**Masque de segmentation**")
            st.image(
                mask_image,
                caption="Vert : animal | Bleu : arrière-plan | Rouge : bordure",
                use_container_width=True
            )

        st.write("**Segmentation superposée à l'image**")
        st.image(
            overlay,
            caption="Visualisation du masque prédit",
            use_container_width=True
        )

        st.caption(
            "Les prédictions sont expérimentales. Une confiance élevée "
            "ne garantit pas que la race soit correcte."
        )

    except Exception as e:
        st.error(f"Une erreur est survenue : {e}")
        st.info(
            "Vérifiez que les deux modèles, le fichier breed_names.json "
            "et les dépendances sont présents dans le dépôt."
        )
