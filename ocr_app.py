import pytesseract
from PIL import Image
import streamlit as st
import io
from pdf2image import convert_from_bytes, convert_from_path
import tempfile
import os
import cv2
import numpy as np

os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def perform_ocr(image):
    # Aumentar a resolução da imagem
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Converter para escala de cinza
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    
    # Aplicar threshold adaptativo
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Remover ruído
    denoised = cv2.fastNlMeansDenoising(binary)
    
    # Configuração específica para português
    custom_config = r'--oem 3 --psm 6 -l por'
    
    # Realizar OCR
    extracted_text = pytesseract.image_to_string(denoised, config=custom_config)
    
    return extracted_text

def process_pdf(pdf_file):
    poppler_path = r"C:\Users\Felipe\AppData\Local\Programs\Python\Python312\Library\bin\poppler-24.08.0\Library\bin"
    
    # Criar um arquivo temporário para salvar o conteúdo do PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_file.getvalue())
        temp_pdf_path = temp_pdf.name

    try:
        images = convert_from_path(temp_pdf_path, poppler_path=poppler_path)
        all_text = []
        for i, image in enumerate(images):
            text = perform_ocr(image)
            all_text.append(f"Página {i+1}:\n{text}\n")
        return "\n".join(all_text)
    finally:
        # Remover o arquivo temporário
        os.unlink(temp_pdf_path)

def main():
    st.title("Conversor de Imagem/PDF para Texto")

    uploaded_file = st.file_uploader("Escolha uma imagem ou PDF", type=["png", "jpg", "jpeg", "pdf"])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()

        if file_extension == "pdf":
            if st.button("Converter PDF para Texto"):
                with st.spinner("Processando PDF..."):
                    text = process_pdf(uploaded_file)
                st.subheader("Texto Extraído:")
                st.text_area("", value=text, height=300)
                
                text_bytes = text.encode('utf-8')
                st.download_button(
                    label="Baixar Texto",
                    data=text_bytes,
                    file_name="texto_extraido.txt",
                    mime="text/plain"
                )
        else:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagem carregada", use_column_width=True)

            if st.button("Converter para Texto"):
                text = perform_ocr(image)
                st.subheader("Texto Extraído:")
                st.text_area("", value=text, height=300)
                
                text_bytes = text.encode('utf-8')
                st.download_button(
                    label="Baixar Texto",
                    data=text_bytes,
                    file_name="texto_extraido.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()

