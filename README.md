# Plataforma de analisis defensivo con tracking y eventing

Aplicacion web en Streamlit para analizar la estructura defensiva de un equipo a partir de datos BePro. El proyecto integra descarga de datos, procesamiento tactico, visualizacion web, autenticacion con Firebase y automatizacion con GitHub Actions.

La interfaz publica anonimiza identidades: el equipo analizado aparece como `Tu Equipo` y los rivales como `Equipo Rival 1`, `Equipo Rival 2`, etc.

## Estructura

```text
app/                         Interfaz web Streamlit.
src/tfg_analysis/            Codigo del pipeline y modulos reutilizables.
scripts/                     Scripts de descarga, regeneracion y automatizacion.
assets/                      Recursos visuales anonimos usados por la app.
eventing_partidos/           CSV de eventos BePro versionados.
sequences_partidos/          CSV de secuencias BePro versionados.
tracking_partidos/           Carpeta local para tracking crudo no versionado.
outputs/app_data/            Datos ligeros e imagenes que consume la app.
outputs/global_clusters/     Modelo y tablas del clustering global.
outputs/memoria_4_3/         Figuras auxiliares usadas en la memoria.
.github/workflows/           Automatizacion de actualizacion semanal.
```

## Por que no se versiona el tracking crudo

Los CSV de tracking pesan mas de 300 MB por partido y GitHub bloquea archivos individuales superiores a 100 MB. Por ese motivo `tracking_partidos/*.csv` esta excluido en `.gitignore`.

El flujo correcto es:

```text
tracking/eventing/sequences BePro
        -> pipeline de analisis
        -> outputs/app_data/
        -> aplicacion web
```

La app desplegada no necesita leer el tracking bruto: usa los datos ligeros ya generados en `outputs/app_data/`. El tracking solo es necesario para recalcular localmente o durante una ejecucion automatica del pipeline.

Si se quiere conservar historico de tracking bruto, debe usarse almacenamiento externo como Drive, S3, Firebase Storage, GitHub Releases o Git LFS.

## Ejecutar la app en local

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH="src"
streamlit run app/streamlit_app.py
```

## Regenerar datos web

Para regenerar todos los partidos que tengan tracking local:

```powershell
$env:PYTHONPATH="src"
python scripts/build_app_data.py --team-id 12987 --force-analysis
```

Para regenerar un unico partido:

```powershell
python scripts/build_app_data.py --team-id 12987 --match-id 199542 --force-analysis
```

## Automatizacion BePro

El script principal es:

```powershell
python scripts/auto_update_bepro.py --team-id 12987
```

Funcionamiento:

1. Lee credenciales BePro desde variables de entorno.
2. Consulta partidos disponibles en BePro.
3. Compara contra los partidos ya generados en `outputs/app_data/`.
4. Si hay un partido nuevo con tracking disponible, descarga tracking, eventing y sequences.
5. Ejecuta el pipeline completo.
6. Genera los datos ligeros de la web.
7. Sube a GitHub los resultados necesarios para la app.

Variables admitidas:

```text
BEPRO_API_KEY       Token BePro recomendado.
BEPRO_DATA_TOKEN    Alternativa al token principal.
BEPRO_AUTH_TOKEN    Alternativa al token principal.
BEPRO_TEAM_ID       ID del equipo propio. Por defecto: 12987.
BEPRO_SEASON_IDS    ID o IDs usados para listar partidos, separados por coma.
```

## GitHub Actions

El workflow esta en:

```text
.github/workflows/update_app_data.yml
```

Se ejecuta cada martes a las 07:00 hora de Madrid y tambien puede lanzarse manualmente desde la pestana `Actions` de GitHub.

El workflow commitea:

```text
outputs/app_data/
eventing_partidos/
sequences_partidos/
outputs/global_clusters/
```

No commitea `tracking_partidos/*.csv` por el limite de tamano de GitHub.

## Firebase

La app usa Firebase para:

```text
Authentication       Login y registro de usuarios.
Firestore usuarios/  Rol, email y equipo asignado.
Firestore equipos/   Equipos detectados por los datos disponibles.
Firestore logs/      Registro basico de acciones.
```

Roles:

```text
admin      Gestiona usuarios y accede a todos los equipos.
analista   Accede solo al equipo asignado.
invitado   Acceso limitado a portada y contenido publico.
```

## Archivos clave

```text
app/streamlit_app.py                  Interfaz principal.
src/tfg_analysis/auth.py              Autenticacion, roles y Firestore.
src/tfg_analysis/app_data.py          Generacion de datos ligeros para la web.
src/tfg_analysis/pipeline/            Pipeline tactico por partido.
src/tfg_analysis/io/bepro.py          Llamadas a la API de BePro.
src/tfg_analysis/sequences/           Deteccion de secuencias rivales.
src/tfg_analysis/visualization/       Graficos y dashboard.
scripts/auto_update_bepro.py          Automatizacion de partidos nuevos.
scripts/build_app_data.py             Regeneracion local de outputs.
```
