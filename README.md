# App de ajuste lineal con Streamlit

Esta aplicación permite:

- capturar pares `x,y` de forma manual,
- cargar archivos CSV o TXT,
- calcular una regresión lineal simple,
- mostrar la gráfica del ajuste,
- mostrar la gráfica de residuales,
- y descargar un reporte PDF.

## Ejecución local

```bash
py -m pip install -r requirements.txt
py -m streamlit run app_streamlit_regresion_lineal_pdf.py
```

## Estructura mínima del repositorio

```text
.
├── app_streamlit_regresion_lineal_pdf.py
├── requirements.txt
└── README.md
```

## Despliegue en Streamlit Community Cloud

1. Suba estos archivos a un repositorio de GitHub.
2. Inicie sesión en `https://share.streamlit.io/`.
3. Conecte su cuenta de GitHub.
4. Cree una nueva app desde su repositorio.
5. Seleccione:
   - Repository: su repositorio
   - Branch: `main`
   - Main file path: `app_streamlit_regresion_lineal_pdf.py`
6. Pulse **Deploy**.
