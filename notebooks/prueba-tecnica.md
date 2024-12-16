# Proyecto 2. CinemAnalytics: Explorando el Cine a través de Datos y Tecnología

## 1. Resumen

Nos ha contratado una plataforma de streaming para mejorar la calidad de su contenido y la experiencia de sus usuarios. Este proyecto tiene como objetivo aplicar técnicas de análisis de datos para identificar las películas y cortometrajes más populares y mejor valorados desde 1990 hasta la fecha, además de desarrollar un sistema de recomendación para personalizar la experiencia de los usuarios.

### Objetivos principales:
- Identificar las películas y cortometrajes más destacados en la plataforma basándose en calificaciones, número de visualizaciones y reseñas.
- Analizar la evolución de las preferencias de los usuarios a lo largo de los años para identificar tendencias clave en la industria cinematográfica.
- Diseñar y desarrollar un sistema de recomendación para sugerir contenido relevante a los usuarios.
- Proporcionar recomendaciones específicas para la promoción de contenido en las distintas secciones de la plataforma.

---

## 2. Fases del Proyecto

### Fase 1: Extracción de Datos de API de MoviesDataset
En esta fase se utilizará la API de MoviesDataset para extraer información relevante sobre películas y cortometrajes.

**Requerimientos:**
- Películas desde 1990 hasta la actualidad.
- Géneros: Drama, Comedy, Action, Fantasy, Horror, Mystery, Romance, Thriller.
- Información necesaria:
  - Tipo (corto o película).
  - Nombre.
  - Año y mes de estreno.
  - ID de la película.

**Nota:** Los datos extraídos deberán almacenarse en una lista de tuplas.

---

### Fase 2: Extracción de Detalles de Películas con Selenium
Utiliza Selenium para obtener información adicional de las películas listadas previamente.

**Información requerida:**
- Calificación de IMDB.
- Dirección (director o directores).
- Guionistas.
- Argumento.
- Duración (en minutos).

**Nota:** Los datos obtenidos deberán almacenarse en una lista de tuplas.

---

### Fase 3: Extracción de Detalles de Actores con Selenium
Obtén información sobre los 10 actores principales de cada película o corto utilizando Selenium.

**Información requerida:**
- Nombre.
- Año de nacimiento.
- Por qué es conocido.
- Roles (actuación, dirección, etc.).
- Premios.

**Nota:** La información deberá almacenarse en una lista de tuplas.

---

### Fase 4: Creación de un Sistema de Recomendación
Diseña y desarrolla un sistema de recomendación basado en los datos recopilados.

**Modelos recomendados:**
- **Modelo colaborativo:** Basado en las preferencias de los usuarios.
- **Modelo basado en contenido:** Considerando géneros, directores y calificaciones.

El sistema debe permitir sugerir películas personalizadas para diferentes tipos de usuarios.

---

### Fase 5: Creación de una Base de Datos
Organiza toda la información recopilada en una base de datos SQL bien estructurada. Define las tablas y relaciones necesarias para almacenar los datos de manera eficiente.

---

### Fase 6: Inserción de Datos en la Base de Datos
Inserta todos los datos recopilados en la base de datos diseñada.

---

### Fase 7: Consultas SQL
Realiza consultas para responder preguntas específicas, como:
- ¿Qué géneros han recibido más premios Oscar?
- ¿Qué género tiene las mejores calificaciones en IMDB?
- ¿Qué director tiene más películas premiadas?
- ¿Cuál es la película mejor valorada en IMDB?
- ¿Qué actor/actriz ha recibido más premios?