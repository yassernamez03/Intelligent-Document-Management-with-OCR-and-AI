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

class Document:
    def __init__(self, name, content, category=None, metadata=None):
        self.name = name
        self.content = content
        self.category = category
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.confidence_score = None

def preprocess_image(image):
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    # Amélioration du contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    # Débruitage
    denoised = cv2.fastNlMeansDenoising(gray)
    return denoised

def extract_text_from_image(image):
    # Prétraitement de l'image
    processed_image = preprocess_image(image)
    # Extraction du texte avec OCR
    text = pytesseract.image_to_string(processed_image, lang='fra+eng')
    return text

def train_classifier(documents):
    if not documents:
        return None
    
    # Création du pipeline de classification
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', MultinomialNB())
    ])
    
    # Préparation des données
    texts = [doc.content for doc in documents if doc.category]
    labels = [doc.category for doc in documents if doc.category]
    
    if len(set(labels)) < 2:
        return None
        
    # Entraînement du modèle
    pipeline.fit(texts, labels)
    return pipeline

def main():
    st.title("Système de Gestion Intelligente des Documents")
    
    # Sidebar pour la navigation
    menu = st.sidebar.selectbox(
        "Menu Principal",
        ["Import et OCR", "Classification", "Analyse et Statistiques", "Recherche", "Export"]
    )
    
    if menu == "Import et OCR":
        st.header("Import et OCR de Documents")
        
        uploaded_file = st.file_uploader("Choisir un document", type=['png', 'jpg', 'jpeg', 'pdf'])
        
        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                # Conversion PDF en images
                images = pdf2image.convert_from_bytes(uploaded_file.read())
                text_content = ""
                for img in images:
                    text_content += extract_text_from_image(img) + "\n"
            else:
                image = Image.open(uploaded_file)
                text_content = extract_text_from_image(image)
            
            st.write("Texte extrait :")
            st.text_area("", text_content, height=300)
            
            # Métadonnées
            st.subheader("Métadonnées")
            col1, col2 = st.columns(2)
            with col1:
                category = st.text_input("Catégorie du document")
            with col2:
                tags = st.text_input("Tags (séparés par des virgules)")
            
            if st.button("Sauvegarder le document"):
                metadata = {
                    "tags": [tag.strip() for tag in tags.split(",")],
                    "file_type": uploaded_file.type,
                    "size": uploaded_file.size
                }
                doc = Document(uploaded_file.name, text_content, category, metadata)
                st.session_state.documents.append(doc)
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