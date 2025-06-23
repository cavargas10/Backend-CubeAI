# Instant3D - Plataforma Web para Generación de Modelos 3D con IA

## Proyecto de Tesis: "Objetos tridimensionales creados por IA: Innovación en entornos virtuales"

Este repositorio contiene el código fuente del servidor (backend) que da vida al proyecto de tesis Instant3D, una plataforma para la generación de modelos 3D mediante Inteligencia Artificial.

**Proyecto desplegado en:** [https://instant3d.vercel.app/](https://instant3d.vercel.app/)

## 🎓 Propósito del Proyecto

Este proyecto tiene como objetivo principal investigar y desarrollar una solución accesible para la creación de contenido 3D, demostrando la innovación que la Inteligencia Artificial puede aportar a los entornos virtuales. Busca facilitar a usuarios, tanto técnicos como no técnicos, la generación de activos 3D a partir de conceptos simples, contribuyendo así al avance y aplicación de la IA en el modelado tridimensional.

## ✨ Funcionalidades Principales

*   **Autenticación y Gestión de Usuarios:** Sistema seguro de registro y perfil de usuario utilizando Firebase.
*   **Múltiples Métodos de Generación 3D con IA:**
    *   **Texto a 3D:** Genera modelos a partir de descripciones textuales.
    *   **Imagen a 3D:** Convierte una imagen 2D en un modelo 3D.
    *   **Texto a Imagen a 3D:** Proceso en dos etapas que primero genera una imagen desde texto y luego la convierte a 3D.
    *   **Múltiples Imágenes a 3D:** Utiliza vistas frontal, lateral y trasera para reconstruir un objeto 3D.
    *   **Boceto a 3D:** Permite dibujar en un lienzo interactivo y convertir el boceto en un modelo 3D.
    *   **Único a 3D:** Método especializado para la generación de objetos 3D a partir de una sola imagen con características particulares.
*   **Galería Personal:** Cada usuario cuenta con una galería personal para visualizar, gestionar y descargar sus creaciones.
*   **Almacenamiento en la Nube:** Uso de Firebase para almacenar de forma persistente tanto los metadatos de las generaciones como los archivos 3D (.glb).

## 🛠️ Tecnologías Utilizadas (Backend)

*   **Backend:** Python, Flask
*   **Base de Datos y Almacenamiento:** Firebase (Firestore para datos y Cloud Storage para archivos)
*   **Autenticación:** Firebase Authentication (Tokens JWT)
*   **Integración de IA:** Hugging Face Spaces a través de la librería gradio_client
*   **Despliegue:** Preparado para la plataforma Koyeb (mediante Procfile)

## 📁 Estructura del Proyecto (Frontend)

La arquitectura del backend está organizada de manera modular para facilitar su mantenimiento y escalabilidad.

```
backendCUBEAI/
├─ config/         # Conexión a servicios externos (Firebase, Hugging Face)
├─ middleware/     # Capas intermedias, como la de autenticación
├─ routes/         # Definición de los endpoints de la API
├─ services/       # Lógica de negocio para cada funcionalidad
├─ utils/          # Funciones de utilidad (ej. subida de archivos)
├─ .env            # Variables de entorno (credenciales)
├─ app.py          # Punto de entrada de la aplicación Flask
├─ requirements.txt# Dependencias de Python
└─ Procfile        # Instrucciones para el despliegue
```

## 🚀 Despliegue

El proyecto está actualmente desplegado en Vercel y accesible en:
[https://instant3d.vercel.app/](https://instant3d.vercel.app/)

## 👨‍💻 Autor

**Carlos Andrés Vargas Ramírez**
*   Estudiante de Ingeniería en Sistemas Informáticos y Computación - Universidad Técnica Particular de Loja (UTPL)

## 📄 Licencia

Este proyecto se presenta como parte del trabajo de tesis **"Objetos tridimensionales creados por IA: Innovación en entornos virtuales"** y su uso está principalmente destinado a fines académicos y de demostración en dicho contexto.