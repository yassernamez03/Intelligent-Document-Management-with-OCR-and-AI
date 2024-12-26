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
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import Counter
import string
from rake_nltk import Rake

# Configuration de la page
st.set_page_config(page_title="Système de Gestion des Documents", layout="wide")

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('averaged_perceptron_tagger')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    
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
if 'suggested_tags' not in st.session_state:
    st.session_state.suggested_tags = [] 
    
     
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

def generate_tags(text, num_tags=10):
    """
    Génère des tags automatiquement en utilisant TF-IDF et RAKE
    """
    if not text:
        return []
    
    tags = set()
    
    # 1. Méthode RAKE
    rake = Rake(language='french')
    rake.extract_keywords_from_text(text)
    rake_phrases = rake.get_ranked_phrases()[:num_tags]
    tags.update(rake_phrases)

    # 2. Méthode TF-IDF pour les mots uniques
    try:
        # Préparation du texte
        stop_words = set(stopwords.words('french') + stopwords.words('english'))
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalnum() and w not in stop_words and len(w) > 2]
        
        # Création d'un petit corpus pour TF-IDF
        vectorizer = TfidfVectorizer(max_features=num_tags, 
                                   stop_words=list(stop_words),
                                   ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([' '.join(words)])
        feature_array = np.array(vectorizer.get_feature_names_out())
        tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
        top_words = feature_array[tfidf_sorting][:num_tags]
        tags.update(top_words)
    except Exception as e:
        st.warning(f"Erreur lors de l'extraction TF-IDF : {str(e)}")

    # 3. Extraction des noms propres et entités importantes
    try:
        tagged_words = pos_tag(words)
        proper_nouns = [word for word, tag in tagged_words if tag.startswith('NNP')]
        tags.update(proper_nouns[:num_tags//2])
    except Exception as e:
        st.warning(f"Erreur lors de l'extraction des noms propres : {str(e)}")

    # Nettoyage et filtrage des tags
    cleaned_tags = []
    for tag in tags:
        # Nettoyage basique
        tag = tag.strip().lower()
        # Filtrage par longueur et contenu
        if len(tag) > 2 and not tag.isdigit() and tag not in stop_words:
            cleaned_tags.append(tag)

    # Tri par longueur et limitation du nombre de tags
    cleaned_tags = sorted(set(cleaned_tags), key=len)[:num_tags]
    return cleaned_tags


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
    st.session_state.suggested_tags = []
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
                    # Section des tags
                    st.subheader("Tags")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        tags = st.text_input("Tags manuels (séparés par des virgules)", 
                                         value=st.session_state.tags,
                                         key='tags_input')
                    
                    with col2:
                        num_tags = st.slider("Nombre de tags à générer", 5, 20, 10)
                    
                    if st.form_submit_button("Générer des tags automatiques"):
                        suggested_tags = generate_tags(st.session_state.text_content, num_tags=num_tags)
                        st.session_state.suggested_tags = suggested_tags
                        if suggested_tags:
                            st.session_state.tags = ', '.join(suggested_tags)
                    
                    # Affichage des tags suggérés s'ils existent
                    if st.session_state.suggested_tags:
                        st.write("Tags suggérés :")
                        selected_tags = []
                        for tag in st.session_state.suggested_tags:
                            if st.checkbox(tag, key=f"tag_{tag}"):
                                selected_tags.append(tag)
                        if selected_tags:
                            st.session_state.tags = ', '.join(selected_tags)
                            
                                        submit_button = st.form_submit_button(label='Sauvegarder le document')
                    if submit_button:
                        if save_document(uploaded_file, 
                                      st.session_state.text_content,
                                      category, 
                                      tags, 
                                      processing_params):
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