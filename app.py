import streamlit as st
import pytesseract
from PIL import Image
import pandas as pd
import numpy as np
from datetime import datetime
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
import pdf2image
import cv2

# Configuration de la page
st.set_page_config(page_title="Système de Gestion des Documents", layout="wide")


# Initialisation des variables de session
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'text_content' not in st.session_state:
    st.session_state.text_content = ""
if 'category' not in st.session_state:
    st.session_state.category = ""
if 'tags' not in st.session_state:
    st.session_state.tags = ""
    
def update_category(value):
    st.session_state.category = value

def update_tags(value):
    st.session_state.tags = value


class Document:
    def __init__(self, name, content, category=None, metadata=None):
        self.name = name
        self.content = content
        self.category = category
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.confidence_score = None

def preprocess_image(image, brightness=0, contrast=1.0, blur_amount=0):
    # Conversion en array numpy
    img_array = np.array(image)
    
    # Ajustement de la luminosité
    img_array = cv2.convertScaleAbs(img_array, alpha=contrast, beta=brightness)
    
    # Application du flou si nécessaire
    if blur_amount > 0:
        img_array = cv2.GaussianBlur(img_array, (blur_amount*2+1, blur_amount*2+1), 0)
    
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Amélioration du contraste adaptatif
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    processed = clahe.apply(gray)
    
    # Débruitage
    denoised = cv2.fastNlMeansDenoising(processed)
    
    return denoised

def extract_text_from_image(image):
    # Extraction du texte avec OCR
    text = pytesseract.image_to_string(image, lang='fra+eng')
    return text

def train_classifier(documents):
    if not documents:
        return None
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', MultinomialNB())
    ])
    
    texts = [doc.content for doc in documents if doc.category]
    labels = [doc.category for doc in documents if doc.category]
    
    if len(set(labels)) < 2:
        return None
        
    pipeline.fit(texts, labels)
    return pipeline

def save_document(file, text_content, category, tags):
    metadata = {
        "tags": [tag.strip() for tag in tags.split(",") if tag.strip()],
        "file_type": file.type,
        "size": file.size}
    
    doc = Document(file.name, text_content, category, metadata)
    st.session_state.documents.append(doc)
    # Réinitialiser les champs après la sauvegarde
    st.session_state.category = ""
    st.session_state.tags = ""
    st.session_state.text_content = ""
    return True


def main():
    st.title("Système de Gestion Intelligente des Documents")
    
    menu = st.sidebar.selectbox(
        "Menu Principal",
        ["Import et OCR", "Classification", "Analyse et Statistiques", "Recherche", "Export"]
    )
    
    if menu == "Import et OCR":
        st.header("Import et OCR de Documents")
        
        uploaded_file = st.file_uploader("Choisir un document", type=['png', 'jpg', 'jpeg', 'pdf'])
        
        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                images = pdf2image.convert_from_bytes(uploaded_file.read())
                st.session_state.current_image = images[0]  # On prend la première page pour commencer
            else:
                st.session_state.current_image = Image.open(uploaded_file)
            
            # Interface de prévisualisation et d'édition
            st.subheader("Prévisualisation et Édition")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(st.session_state.current_image, caption="Image originale", use_column_width=True)
            
            with col2:
                st.subheader("Paramètres de traitement")
                brightness = st.slider("Luminosité", -100, 100, 0)
                contrast = st.slider("Contraste", 0.0, 3.0, 1.0, 0.1)
                blur = st.slider("Flou", 0, 5, 0)
                
                if st.button("Appliquer le traitement"):
                    processed = preprocess_image(
                        st.session_state.current_image,
                        brightness=brightness,
                        contrast=contrast,
                        blur_amount=blur
                    )
                    st.session_state.processed_image = processed
                    st.image(processed, caption="Image traitée", use_column_width=True)
            
            # Extraction du texte
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Extraire le texte"):
                    if st.session_state.processed_image is not None:
                        st.session_state.text_content = extract_text_from_image(st.session_state.processed_image)
                    else:
                        st.session_state.text_content = extract_text_from_image(
                            preprocess_image(st.session_state.current_image))
            
            if st.session_state.text_content:
                st.text_area("Texte extrait :", st.session_state.text_content, height=300)
                
                # Formulaire de métadonnées
                with st.form(key='metadata_form'):
                    st.subheader("Métadonnées")
                    category = st.text_input("Catégorie du document", 
                                          value=st.session_state.category,
                                          key='category_input')
                    tags = st.text_input("Tags (séparés par des virgules)", 
                                     value=st.session_state.tags,
                                     key='tags_input')
                    
                    submit_button = st.form_submit_button(label='Sauvegarder le document')
                    if submit_button:
                        if save_document(uploaded_file, 
                                      st.session_state.text_content,
                                      category, 
                                      tags):
                            st.success(f"Document '{uploaded_file.name}' sauvegardé avec succès!")

    elif menu == "Classification":
        st.header("Classification des Documents")
        
        if st.button("Entraîner le classificateur"):
            classifier = train_classifier(st.session_state.documents)
            if classifier:
                st.session_state.classifier = classifier
                st.success("Classificateur entraîné avec succès!")
            else:
                st.warning("Pas assez de données pour entraîner le classificateur")
        
        if st.session_state.classifier:
            st.subheader("Classification automatique")
            text_to_classify = st.text_area("Entrez le texte à classifier")
            if st.button("Classifier"):
                prediction = st.session_state.classifier.predict([text_to_classify])[0]
                proba = st.session_state.classifier.predict_proba([text_to_classify])[0]
                st.write(f"Catégorie prédite : {prediction}")
                st.write(f"Score de confiance : {max(proba):.2%}")
    
    elif menu == "Analyse et Statistiques":
        st.header("Analyse et Statistiques")
        
        if st.session_state.documents:
            # Distribution des catégories
            categories = [doc.category for doc in st.session_state.documents if doc.category]
            if categories:
                fig, ax = plt.subplots()
                sns.countplot(y=categories)
                plt.title("Distribution des catégories de documents")
                st.pyplot(fig)
            
            # Timeline des uploads
            dates = [doc.timestamp for doc in st.session_state.documents]
            fig, ax = plt.subplots()
            plt.hist(dates, bins=20)
            plt.title("Timeline des documents importés")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # Statistiques textuelles
            st.subheader("Statistiques textuelles")
            word_counts = [len(doc.content.split()) for doc in st.session_state.documents]
            st.write(f"Nombre moyen de mots par document : {np.mean(word_counts):.0f}")
            st.write(f"Nombre total de documents : {len(st.session_state.documents)}")
    
    elif menu == "Recherche":
        st.header("Recherche de Documents")
        
        search_query = st.text_input("Rechercher dans les documents")
        if search_query:
            results = []
            for doc in st.session_state.documents:
                if search_query.lower() in doc.content.lower():
                    results.append(doc)
            
            st.write(f"Résultats trouvés : {len(results)}")
            for doc in results:
                with st.expander(f"{doc.name} - {doc.category}"):
                    st.write(doc.content[:500] + "...")
                    st.write("Métadonnées:", doc.metadata)
    
    elif menu == "Export":
        st.header("Export des Données")
        
        if st.session_state.documents:
            export_format = st.selectbox("Format d'export", ["CSV", "JSON", "Excel"])
            
            if st.button("Exporter"):
                data = []
                for doc in st.session_state.documents:
                    data.append({
                        "nom": doc.name,
                        "categorie": doc.category,
                        "contenu": doc.content[:1000],
                        "date": doc.timestamp,
                        "metadata": str(doc.metadata)
                    })
                
                df = pd.DataFrame(data)
                
                if export_format == "CSV":
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Télécharger CSV",
                        csv,
                        "documents_export.csv",
                        "text/csv"
                    )
                elif export_format == "JSON":
                    json = df.to_json(orient="records")
                    st.download_button(
                        "Télécharger JSON",
                        json,
                        "documents_export.json",
                        "application/json"
                    )
                else:  # Excel
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, sheet_name='Documents', index=False)
                    st.download_button(
                        "Télécharger Excel",
                        output.getvalue(),
                        "documents_export.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

if __name__ == "__main__":
    main()